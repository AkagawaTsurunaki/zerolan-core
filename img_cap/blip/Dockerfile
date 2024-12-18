# Get pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

LABEL name="zerolan-core-imgcap-blip" \
    maintainer="AkagawaTsurunaki <AkagawaTsurunaki@outlook.com>" \
    lastupdate="2024-11-06"

# Create workspace folder and set it as working directory
RUN mkdir -p /workspace/services/ && \
    mkdir -p /workspace/models/
WORKDIR /workspace

# Set the timezone
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y tzdata && \
    ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata

# Install git
RUN apt-get install git -y && apt-get install git-lfs
RUN apt-get install wget

# Mirror repo from: https://huggingface.co/Salesforce/blip-image-captioning-large
RUN cd /workspace/models/ && \
    git lfs install && \
    git clone https://gitee.com/hf-models/blip-image-captioning-large.git
    
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
RUN conda create -n blip python=3.10 -y && echo "source activate blip" > ~/.bashrc
ENV PATH /workspace/miniconda3/envs/blip/bin:$PATH

# Clone the repository
RUN cd /workspace/services/ && \
    git clone https://github.com/AkagawaTsurunaki/zerolan-core.git
    
# [Dev] Check to branch dev
RUN cd /workspace/services/zerolan-core && \ 
    git checkout dev

# Install required dependencies

# NOTE: If you need these mirrors for pip, please uncomment them
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip config set global.index-url https://pypi.mirrors.ustc.edu.cn/simple && \
    pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# NOTE: If you need these mirror for torch and torchvision, please uncomment it
RUN pip install torch==2.3.1 torchvision -f https://mirrors.aliyun.com/pytorch-wheels/cu121/
# RUN pip install torch==2.3.1 torchvision --index-url https://download.pytorch.org/whl/cu121/

# [Dev] Install zerolan-data
RUN pip install git+https://github.com/AkagawaTsurunaki/zerolan-data.git@dev
RUN cd /workspace/services/zerolan-core && \
    pip install -r requirements.txt && \
    pip install -r ./img_cap/blip/requirements.txt

# Copy config file
RUN cd /workspace/services/zerolan-core && \
    cp config.template.yaml config.yaml
