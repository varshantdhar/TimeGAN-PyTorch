from __future__ import division
from matplotlib.pyplot import axis
import numpy as np

def batch_iterator(X, y=None, batch_size=64):
    n_samples = X.shape[0]
    for i in np.arange(0, n_samples, batch_size):
        begin, end = i, min(i+batch_size, n_samples)
        if y is not None:
            yield X[begin:end], y[begin:end]
        else:
            yield X[int(begin):int(end)]

def accuracy_score(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy

def shuffle_data(X, seed=None):
    if seed:
        np.random.seed(seed)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx]

def shuffle_both(X, y, seed=None):
    if seed:
        np.random.seed(seed)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]

def train_test_split(X, y, test_size=0.5, shuffle=True, seed=None):
    if shuffle:
        X, y = shuffle_data(X, y, seed)
    # Split the training data from test data in the ratio specified in
    # test_size
    split_i = len(y) - int(len(y) // (1 / test_size))
    X_train, X_test = X[:split_i], X[split_i:]
    y_train, y_test = y[:split_i], y[split_i:]

    return X_train, X_test, y_train, y_test

def merge(dat, merge_mode='ave'):
    if merge_mode == 'ave':
        return np.mean(np.array([dat[:,:,:,0], dat[:,:,:,1]]), axis=0)
    elif merge_mode == 'sum':
        return np.sum(np.array([dat[:,:,:,0], dat[:,:,:,1]]), axis=0)
    elif merge_mode == 'mul':
        return np.multiply(np.array([dat[:,:,:,0], dat[:,:,:,1]]), axis=0)
    elif merge_mode == 'concat':
        return dat.reshape((dat.shape[0], dat.shape[1], dat.shape[2] * 2))
    else:
        raise ValueError('merge_mode not found')

def MinMaxScaler(data): 
    min_val = np.min(np.min(data, axis = 0), axis = 0)
    data = data - min_val
    
    max_val = np.max(np.max(data, axis = 0), axis = 0)
    norm_data = data / (max_val + 1e-7)
    
    return norm_data, min_val, max_val

def StandardScaler(data):
    return np.apply_along_axis(lambda x: (x - np.mean(x)) / np.std(x), 2, data)

def random_generator (batch_size, z_dim, T_mb, max_seq_len):
  Z_mb = list()
  for i in range(batch_size):
    temp = np.zeros([max_seq_len, z_dim])
    temp_Z = np.random.uniform(0., 1, size=(T_mb[i], z_dim))
    temp[:T_mb[i],:] = temp_Z
    Z_mb.append(temp_Z)
  return Z_mb

def pad_along_axis(array: np.ndarray, target_length: int, axis: int = 0):

    pad_size = target_length - array.shape[axis]

    if pad_size <= 0:
        return array

    npad = [(0, 0)] * array.ndim
    npad[axis] = (0, pad_size)

    return np.pad(array, pad_width=npad, mode='constant', constant_values=0)
    
