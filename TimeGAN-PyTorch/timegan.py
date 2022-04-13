import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pickle
import wandb

from unicodedata import bidirectional
from utils import random_generator, batch_iterator
from google.cloud import storage
from tqdm import tqdm
from time import sleep
from config import WANDB_API_KEY, WANDB_ENTITY, WANDB_PROJECT, BUCKET_NAME

storage_client = storage.Client()

bucket = storage_client.bucket(BUCKET_NAME)

class AutoEncoder(nn.Module):
    def __init__(self, input_size, num_layers, dropout):
        super().__init__()
        self.hidden_size = 200
        self.embedder = nn.LSTM(input_size=input_size, hidden_size=self.hidden_size, num_layers=num_layers,
                                batch_first=True, dropout=dropout)
        self.recovery = nn.Linear(in_features=self.hidden_size, out_features=input_size)

    
    def forward(self, X):
        H, _= self.embedder.forward(X)
        X_tilde = self.recovery.forward(H)
        return X_tilde

class Supervisor(nn.Module):
    def __init__(self, autoencoder, num_layers, dropout, z_dim):
        super().__init__()
        self.hidden_size = 200
        self.embedder = autoencoder.module.embedder
        self.generator = nn.LSTM(input_size=z_dim, hidden_size=self.hidden_size, num_layers=num_layers,
                                batch_first=True, dropout=dropout)
        self.supervisor = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=num_layers,
                                 batch_first=True, dropout=dropout, bidirectional=True)
        
    
    def forward(self, X):
        H ,_  = self.embedder.forward(X)
        H_hat_supervise, _ = self.supervisor.forward(H)
        dim = H.shape[2]
        H_hat_supervise = torch.mean(torch.stack([H_hat_supervise[:,:,:dim], H_hat_supervise[:,:,dim:]]), dim=0)
        return (H, H_hat_supervise)
        
class EmbedderGenerator(nn.Module):
    def __init__(self, autoencoder, supervisor) -> None:
        super().__init__()
        self.embedder = autoencoder.module.embedder
        self.recovery = autoencoder.module.recovery
        self.generator = supervisor.module.generator
        self.supervisor = supervisor.module.supervisor
        self.hidden_size = 200
        self.gamma = 1
        self.discriminator_layers = [nn.Linear(in_features=self.hidden_size, out_features=64),
                                     nn.Dropout(p=0.2),
                                     nn.Linear(in_features=64, out_features=16),
                                     nn.Dropout(p=0.1),
                                     nn.Linear(in_features=16, out_features=4),
                                     nn.Dropout(p=0.1),
                                     nn.Linear(in_features=4, out_features=1),
                                     nn.Sigmoid()]
        self.discriminator = nn.Sequential(*self.discriminator_layers)
    
    def forward(self, X, Z):
        H, _ = self.embedder.forward(X)
        dim = H.shape[2]
        X_tilde = self.recovery.forward(H)

        E_hat, _ = self.generator.forward(Z)
        H_hat, _ = self.supervisor.forward(E_hat)
        H_hat = torch.mean(torch.stack([H_hat[:,:,:dim], H_hat[:,:,dim:]]), dim=0)
        H_hat_supervise, _ = self.supervisor.forward(H)
        H_hat_supervise = torch.mean(torch.stack([H_hat_supervise[:,:,:dim], H_hat_supervise[:,:,dim:]]), dim=0)
        X_hat = self.recovery.forward(H_hat)
        Y_fake = self.discriminator.forward(H_hat)
        Y_fake_e = self.discriminator.forward(E_hat)

        return (X_tilde, X_hat, H, H_hat_supervise, Y_fake, Y_fake_e)
        

class Discriminate(nn.Module):
    def __init__(self, embedder_generator):
        super().__init__()
        self.hidden_size = 200
        self.gamma = 1
        self.embedder = embedder_generator.module.embedder
        self.recovery = embedder_generator.module.recovery
        self.generator = embedder_generator.module.generator
        self.supervisor = embedder_generator.module.supervisor
        self.discriminator = embedder_generator.module.discriminator
    
    def forward(self, X, Z):
        H, _ = self.embedder.forward(X)
        dim = H.shape[2]
        E_hat, _ = self.generator.forward(Z)
        H_hat, _ = self.supervisor.forward(E_hat)
        H_hat = torch.mean(torch.stack([H_hat[:,:,:dim], H_hat[:,:,dim:]]), dim=0)

        Y_real = self.discriminator.forward(H)
        Y_fake = self.discriminator.forward(H_hat)
        Y_fake_e = self.discriminator.forward(E_hat)
        
        return (Y_real, Y_fake, Y_fake_e)

class DataGen(nn.Module):
    def __init__(self, discriminate):
        super().__init__()
        self.generator = discriminate.module.generator
        self.supervisor = discriminate.module.supervisor
        self.recovery = discriminate.module.recovery
        self.hidden_size = 200

    def forward(self, Z):
        E_hat, _ = self.generator.forward(Z)
        H_hat, _ = self.supervisor.forward(E_hat)
        dim = self.hidden_size
        H_hat = torch.mean(torch.stack([H_hat[:,:,:dim], H_hat[:,:,dim:]]), dim=0)
        X_hat = self.recovery.forward(H_hat)
        return X_hat

def main():
    ori_data = pickle.loads(bucket.blob('data.pkl').download_as_string())
    print("Data Loaded")
    print(ori_data.shape)

    def MinMaxScaler(data):  
        min_val = np.min(np.min(data, axis = 0), axis = 0)
        data = data - min_val
        
        max_val = np.max(np.max(data, axis = 0), axis = 0)
        norm_data = data / (max_val + 1e-7)
        
        return norm_data, min_val, max_val
  
    # Normalization
    data, min_val, max_val = MinMaxScaler(ori_data)

    no, seq_len, dim = data.shape
    ori_time = [seq_len] * no
    max_seq_len = seq_len
    X = data.copy()
    z_dim = 120
    num_layers = 4
    dropout = 0.2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.login(key=WANDB_API_KEY)
    wandb.init(entity=WANDB_ENTITY,project=WANDB_PROJECT)

    n_epochs = 500
    batch_size = 512
    wandb.config = {"epochs": n_epochs,"batch_size": batch_size,"learning_rate":1e-4}

    MSEloss = nn.MSELoss()
    CELoss = nn.CrossEntropyLoss()

    autoencoder = AutoEncoder(input_size=dim, num_layers=num_layers, dropout=dropout)
    autoencoder = nn.DataParallel(autoencoder)
    autoencoder.to(device)
    autoencoder.module.embedder = pickle.loads(bucket.blob('model_torch/embedder.pkl').download_as_string())
    autoencoder.module.recovery = pickle.loads(bucket.blob('model_torch/recovery.pkl').download_as_string())
    X = torch.from_numpy(X).float()

    wandb.watch(autoencoder.module.embedder, log_freq=100)
    wandb.watch(autoencoder.module.recovery, log_freq=100)
    E_optimizer = optim.Adam(autoencoder.module.embedder.parameters(), lr=1e-4)
    R_optimizer = optim.Adam(autoencoder.module.recovery.parameters(), lr=1e-4)
    
    for epoch in tqdm(range(n_epochs)):
        for X_batch in batch_iterator(X, batch_size=batch_size):
            input = X_batch.to(device)
            E_optimizer.zero_grad()
            R_optimizer.zero_grad()
            X_tilde = autoencoder(input)
            loss = MSEloss(X_tilde, input)
            loss.backward()
            R_optimizer.step()
            E_optimizer.step()
            wandb.log({"Embedder Loss": float(loss)})
        
        if epoch % 10 == 0:
            bucket.blob('model_torch/embedder.pkl').upload_from_string(pickle.dumps(autoencoder.module.embedder))
            bucket.blob('model_torch/recovery.pkl').upload_from_string(pickle.dumps(autoencoder.module.recovery))
    
    supervisor = Supervisor(autoencoder=autoencoder,num_layers=num_layers, dropout=dropout, z_dim=z_dim)
    supervisor = nn.DataParallel(supervisor)
    supervisor.to(device)
    supervisor.module.generator = pickle.loads(bucket.blob('model_torch/generator.pkl').download_as_string())
    supervisor.module.supervisor = pickle.loads(bucket.blob('model_torch/supervisor.pkl').download_as_string())
    
    wandb.watch(supervisor.module.generator, log_freq=100)
    wandb.watch(supervisor.module.supervisor, log_freq=100)
    G_optimizer = optim.Adam(supervisor.module.generator.parameters(), lr=1e-4)
    S_optimizer = optim.Adam(supervisor.module.supervisor.parameters(), lr=1e-4)
     
    for epoch in tqdm(range(n_epochs)):
        for X_batch in batch_iterator(X, batch_size=batch_size):
            input = X_batch.to(device)
            G_optimizer.zero_grad()
            S_optimizer.zero_grad()
            X_batch.to(device)
            H, H_hat_supervise = supervisor(input)
            loss = MSEloss(H, H_hat_supervise)
            loss.backward()
            G_optimizer.step()
            S_optimizer.step()
            wandb.log({"Supervised Loss": float(loss)})
        
        if epoch % 10 == 0:
            bucket.blob('model_torch/generator.pkl').upload_from_string(pickle.dumps(supervisor.module.generator))
            bucket.blob('model_torch/supervisor.pkl').upload_from_string(pickle.dumps(supervisor.module.supervisor))

    embedder_generator = EmbedderGenerator(autoencoder=autoencoder, supervisor=supervisor)
    embedder_generator = nn.DataParallel(embedder_generator)
    embedder_generator.to(device)
    embedder_generator.module.discriminator = pickle.loads(bucket.blob('model_torch/discriminator.pkl').download_as_string())
    discriminate = Discriminate(embedder_generator=embedder_generator)
    discriminate = nn.DataParallel(discriminate)
    discriminate.to(device)

    D_optimizer = optim.Adam(embedder_generator.module.discriminator.parameters(), lr=1e-4)
    wandb.watch(discriminate.module.discriminator, log_freq=100)
    for epoch in tqdm(range(n_epochs)):
        for _ in range(2):
            for X_batch in batch_iterator(X, batch_size=batch_size):
                Z = random_generator(X_batch.shape[0], z_dim, ori_time, max_seq_len)
                Z = torch.tensor(np.array(Z)).float()
                input = X_batch.to(device)
                noise = Z.to(device)
                G_optimizer.zero_grad()
                E_optimizer.zero_grad()
                R_optimizer.zero_grad()
                S_optimizer.zero_grad()
                X_tilde, X_hat, H, H_hat_supervise, Y_fake, Y_fake_e = embedder_generator(input, noise)

                G_loss_S = MSEloss(H, H_hat_supervise)
                G_loss_U = CELoss(Y_fake, torch.ones_like(Y_fake))
                G_loss_U_e = CELoss(Y_fake_e, torch.ones_like(Y_fake_e))

                G_loss_V1 = torch.mean(torch.abs(torch.sqrt(torch.var_mean(X_hat, 0)[0] + 1e-6) - torch.sqrt(torch.var_mean(input,0)[0] + 1e-6)))
                G_loss_V2 = torch.mean(torch.abs((torch.var_mean(X_hat,0)[1]) - (torch.var_mean(input,0)[1])))
                G_loss_V = G_loss_V1 + G_loss_V2
                g_loss = G_loss_U + G_loss_U_e + 100 * torch.sqrt(G_loss_S) + 100 * G_loss_V

                e_loss = MSEloss(X_tilde, input)

                g_loss.backward(retain_graph=True)
                S_optimizer.step()
                G_optimizer.step()
                e_loss.backward()
                R_optimizer.step()
                E_optimizer.step()
                wandb.log({"Generator Loss": float(g_loss)})
                wandb.log({"Embedding Loss": float(e_loss)})
        for X_batch in batch_iterator(X, batch_size=batch_size):
            Z = random_generator(X_batch.shape[0], z_dim, ori_time, max_seq_len)
            Z = torch.tensor(np.array(Z)).float()
            D_optimizer.zero_grad()
            input = X_batch.to(device)
            noise = Z.to(device)
            Y_real, Y_fake, Y_fake_e = discriminate(input, noise)

            D_loss_real = CELoss(Y_real, torch.ones_like(Y_real))
            D_loss_fake = CELoss(Y_fake, torch.zeros_like(Y_fake))
            D_loss_fake_e = CELoss(Y_fake_e, torch.zeros_like(Y_fake_e))
            d_loss = D_loss_real + D_loss_fake + D_loss_fake_e

            if d_loss > 0.15:
                d_loss.backward()
                D_optimizer.step()
                wandb.log({"Discriminator Loss": float(d_loss)})
        
        if epoch % 10 == 0:
            bucket.blob('model_torch/embedder.pkl').upload_from_string(pickle.dumps(embedder_generator.module.embedder))
            bucket.blob('model_torch/recovery.pkl').upload_from_string(pickle.dumps(embedder_generator.module.recovery))
            bucket.blob('model_torch/generator.pkl').upload_from_string(pickle.dumps(embedder_generator.module.generator))
            bucket.blob('model_torch/supervisor.pkl').upload_from_string(pickle.dumps(embedder_generator.module.supervisor))
            bucket.blob('model_torch/discriminator.pkl').upload_from_string(pickle.dumps(discriminate.module.discriminator))
    
    datagen = DataGen(discriminate=discriminate)
    datagen = nn.DataParallel(datagen)
    datagen.to(device)
    for i in tqdm(range(100,200,1)):
        Z = random_generator(no, z_dim, ori_time, max_seq_len)
        Z = torch.tensor(np.array(Z)).float()
        noise = Z.to(device)
        X_hat = datagen(noise)
        X_hat = X_hat.cpu().detach().numpy()
        # Renormalization
        X_hat *= max_val
        X_hat += min_val
        bucket.blob("synthetic_data/synthetic_set_" + str(i) + ".pkl").upload_from_string(pickle.dumps(X_hat))
        sleep(1/100)
    
class MyDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        return getattr(self.module, name)

if __name__ == "__main__":
    main()


