# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG CUDA_VERSION=11.2.2
ARG CUDNN_VERSION=8
ARG OS_VERSION=20.04

# 从nvidia 官方镜像库拉取基础镜像
FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu${OS_VERSION}
MAINTAINER Wenhui Zhou

# ENV TRT_VERSION 7.2.3.4
# ENV TRT_VERSION 7.0.0.11
ENV TRT_VERSION 8.2.1.8

ENV DEBIAN_FRONTEND=noninteractive
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
    vim \
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

RUN dpkg -i nv-tensorrt-repo-ubuntu1604-cuda10.2-trt8.0.1.6-ga-20210626_1-1_amd64.deb \
cd /var/nv-tensorrt-repo-ubuntu1604-cuda10.2-trt8.0.1.6-ga-20210626 \
dpkg -i libcudnn8_8.2.1.32-1+cuda10.2_amd64.deb \
dpkg -i libcudnn8-dev_8.2.1.32-1+cuda10.2_amd64.deb \
dpkg -i libnvinfer8_8.0.1-1+cuda10.2_amd64.deb \
dpkg -i libnvinfer-dev_8.0.1-1+cuda10.2_amd64.deb \
dpkg -i libnvinfer-plugin8_8.0.1-1+cuda10.2_amd64.deb \
dpkg -i libnvinfer-plugin-dev_8.0.1-1+cuda10.2_amd64.deb \
dpkg -i libnvonnxparsers8_8.0.1-1+cuda10.2_amd64.deb \
dpkg -i libnvonnxparsers-dev_8.0.1-1+cuda10.2_amd64.deb \
dpkg -i onnx-graphsurgeon_8.0.1-1+cuda10.2_amd64.deb \
dpkg -i libnvparsers8_8.0.1-1+cuda10.2_amd64.deb \
dpkg -i libnvparsers-dev_8.0.1-1+cuda10.2_amd64.deb \
dpkg -i libnvinfer-bin_8.0.1-1+cuda10.2_amd64.deb \
dpkg -i python3-libnvinfer_8.0.1-1+cuda10.2_amd64.deb \
dpkg -i python3-libnvinfer-dev_8.0.1-1+cuda10.2_amd64.deb \
dpkg -i libnvinfer-samples_8.0.1-1+cuda10.2_all.deb \
dpkg -i libnvinfer-doc_8.0.1-1+cuda10.2_all.deb \
dpkg -i tensorrt_8.0.1.6-1+cuda10.2_amd64.deb

# 安装 python3 环境
RUN apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    python3-wheel &&\
    cd /usr/local/bin &&\
    ln -s /usr/bin/python3 python &&\
    ln -s /usr/bin/pip3 pip; \



# 安装 TensorRT
RUN cd /tmp && sudo apt-get update

RUN version="8.2.1-1+cuda11.2" && \
    sudo apt-get install libnvinfer8=${version} libnvonnxparsers8=${version} libnvparsers8=${version} libnvinfer-plugin8=${version} libnvinfer-dev=${version} libnvonnxparsers-dev=${version} libnvparsers-dev=${version} libnvinfer-plugin-dev=${version} python3-libnvinfer=${version} &&\
    sudo apt-mark hold libnvinfer8 libnvonnxparsers8 libnvparsers8 libnvinfer-plugin8 libnvinfer-dev libnvonnxparsers-dev libnvparsers-dev libnvinfer-plugin-dev python3-libnvinfer

# 升级 pip 并切换成国内豆瓣源
RUN python3 -m pip install -i https://pypi.douban.com/simple/ --upgrade pip
RUN pip3 config set global.index-url https://pypi.douban.com/simple/
RUN pip3 install setuptools>=41.0.0

# 升级 Cmake（可选）
RUN cd /tmp && \
    wget https://github.com/Kitware/CMake/releases/download/v3.14.4/cmake-3.14.4-Linux-x86_64.sh && \
    chmod +x cmake-3.14.4-Linux-x86_64.sh && \
    ./cmake-3.14.4-Linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license && \
    rm ./cmake-3.14.4-Linux-x86_64.sh

# 设置环境变量和工作路径
ENV TRT_LIBPATH /usr/lib/x86_64-linux-gnu
ENV TRT_OSSPATH /workspace/TensorRT
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${TRT_OSSPATH}/build/out:${TRT_LIBPATH}"
WORKDIR /workspace

RUN ["/bin/bash"]