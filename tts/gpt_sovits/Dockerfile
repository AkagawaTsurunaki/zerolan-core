FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu20.04

LABEL name="zerolan-core-tts-gptsovits" \
    maintainer="AkagawaTsurunaki <AkagawaTsurunaki@outlook.com>" \
    lastupdate="2024-11-06"

# Create workspace folder and set it as working directory
RUN mkdir -p /workspace/services/ && \
    mkdir -p /workspace/models/

# Set the timezone
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y tzdata && \
    ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata

# Install git
RUN apt-get install git -y && apt-get install git-lfs -y
RUN apt-get install wget

# Install conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x Miniconda3-latest-Linux-x86_64.sh && \
    ./Miniconda3-latest-Linux-x86_64.sh -b -p /workspace/miniconda3 && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Update PATH environment variable
ENV PATH="/workspace/miniconda3/bin:${PATH}"

# Initialize conda
RUN conda init bash

# Create and activate conda environment
RUN conda create -n GPTSoVits python=3.10 -y && echo "source activate GPTSoVits" > ~/.bashrc
ENV PATH /workspace/miniconda3/envs/GPTSoVits/bin:$PATH

RUN apt install ffmpeg -y
RUN apt install libsox-dev -y 
RUN conda install -c conda-forge 'ffmpeg<7' -y

WORKDIR /workspace/services
RUN git clone https://github.com/AkagawaTsurunaki/GPT-SoVITS.git

WORKDIR /workspace/services/GPT-SoVITS
RUN python api.py

RUN cd GPT_SoVITS/pretrained_models && \
    git lfs install && \
    git clone https://huggingface.co/lj1995/GPT-SoVITS