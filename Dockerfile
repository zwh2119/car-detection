
ARG CUDA_VERSION=10.2
ARG CUDNN_VERSION=8
ARG OS_VERSION=18.04

FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu${OS_VERSION}
MAINTAINER Wenhui Zhou

# ENV TRT_VERSION 7.2.3.4
ENV TRT_VERSION 8.2.1.8
SHELL ["/bin/bash", "-c"]


# 将 apt 的升级源切换成 阿里云
RUN  sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list && \
            apt-get clean && \
            rm /etc/apt/sources.list.d/*

# 安装必要的库
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4-openssl-dev \
    wget \
    zlib1g-dev \
    git \
    pkg-config \
    sudo \
    ssh \
    libssl-dev \
    pbzip2 \
    pv \
    bzip2 \
    unzip \
    devscripts \
    lintian \
    fakeroot \
    dh-make \
    build-essential \
    libgl1-mesa-glx

# 安装 python3 环境
RUN apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    python3-wheel &&\
    cd /usr/local/bin &&\
    ln -s /usr/bin/python3 python &&\
    ln -s /usr/bin/pip3 pip;

# 安装 TensorRT
RUN cd /tmp &&\
    wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb &&\
    dpkg -i nvidia-machine-learning-repo-*.deb && apt-get update
RUN v="${TRT_VERSION%.*}-1+cuda${CUDA_VERSION%.*}" &&\
    apt-get install -y libnvinfer7=${v} libnvinfer-plugin7=${v} libnvparsers7=${v} libnvonnxparsers7=${v} libnvinfer-dev=${v} libnvinfer-plugin-dev=${v} libnvparsers-dev=${v} python3-libnvinfer=${v} &&\
    apt-mark hold libnvinfer7 libnvinfer-plugin7 libnvparsers7 libnvonnxparsers7 libnvinfer-dev libnvinfer-plugin-dev libnvparsers-dev python3-libnvinfer

# 升级 pip 并切换成国内豆瓣源
RUN python3 -m pip install -i https://pypi.douban.com/simple/ --upgrade pip
RUN pip3 config set global.index-url https://pypi.douban.com/simple/
RUN pip3 install setuptools>=41.0.0



# 设置环境变量和工作路径
ENV TRT_LIBPATH /usr/lib/x86_64-linux-gnu
ENV TRT_OSSPATH /workspace/TensorRT
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${TRT_OSSPATH}/build/out:${TRT_LIBPATH}"
WORKDIR /workspace


RUN ["/bin/bash"]
