FROM nvidia/cuda:11.4.2-cudnn8-devel-ubuntu20.04

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    python3-wheel \
    python3-setuptools && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN pip3 install --no-cache-dir -U install setuptools pip

WORKDIR /root

COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt

COPY timegan.py ./timegan.py
COPY utils.py ./utils.py
ENV WANDB_API_KEY='7de1aa7bce58011a27b011e0ee674e237bb17b53'
ENV WANDB_ENTITY='varshantdhar'
ENV WANDB_PROJECT='TimeGAN'

ENTRYPOINT ["python3","timegan.py"]
