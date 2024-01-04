

ARG CUDA_VERSION=10.2

# Multi-arch container support available in non-cudnn containers.
FROM  samuelwei/cuda10.2-pytorch1.5:laste

ENV TRT_VERSION 8.6.1.6
SHELL ["/bin/bash", "-c"]

# Setup user account
ARG uid=1000
ARG gid=1000
RUN groupadd -r -f -g ${gid} trtuser && useradd -o -r -l -u ${uid} -g ${gid} -ms /bin/bash trtuser
RUN usermod -aG sudo trtuser
RUN echo 'trtuser:nvidia' | chpasswd
RUN mkdir -p /workspace && chown trtuser /workspace

# Required to build Ubuntu 20.04 without user prompts with DLFW container
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y gnupg

# Update CUDA signing key
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/sbsa/3bf863cc.pub

# Install requried libraries
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4-openssl-dev \
    wget \
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
    build-essential

RUN update-binfmts --enable

RUN     dpkg -i libnvinfer8_8.2.1-1+cuda10.2_arm64.deb  \
     && dpkg -i libnvinfer-bin_8.2.1-1+cuda10.2_arm64.deb \
     && dpkg -i libnvinfer-dev_8.2.1-1+cuda10.2_arm64.deb \
     && dpkg -i libnvinfer-doc_8.2.1-1+cuda10.2_all.deb \
     && dpkg -i libnvinfer-plugin8_8.2.1-1+cuda10.2_arm64.deb \
     && dpkg -i libnvinfer-plugin-dev_8.2.1-1+cuda10.2_arm64.deb \
     && dpkg -i libnvinfer-samples_8.2.1-1+cuda10.2_all.deb  \
     && dpkg -i libnvonnxparsers8_8.2.1-1+cuda10.2_arm64.deb \
     && dpkg -i libnvonnxparsers-dev_8.2.1-1+cuda10.2_arm64.deb \
     && dpkg -i libnvparsers8_8.2.1-1+cuda10.2_arm64.deb   \
     && dpkg -i libnvparsers-dev_8.2.1-1+cuda10.2_arm64.deb \
     && dpkg -i graphsurgeon-tf_8.2.1-1+cuda10.2_arm64.deb \
     && dpkg -i uff-converter-tf_8.2.1-1+cuda10.2_arm64.deb \
     && dpkg -i python3-libnvinfer_8.2.1-1+cuda10.2_arm64.deb \
     && dpkg -i python3-libnvinfer-dev_8.2.1-1+cuda10.2_arm64.deb \
     && dpkg -i tensorrt_8.2.1.9-1+cuda10.2_arm64.deb \
     && dpkg -i OpenCV-4.1.1-2-gd5a58aa75-aarch64-dev.deb \
     && dpkg -i OpenCV-4.1.1-2-gd5a58aa75-aarch64-libs.deb \
     && dpkg -i OpenCV-4.1.1-2-gd5a58aa75-aarch64-licenses.deb \
     && dpkg -i OpenCV-4.1.1-2-gd5a58aa75-aarch64-python.deb \
     && dpkg -i OpenCV-4.1.1-2-gd5a58aa75-aarch64-samples.deb  \
     && dpkg -i OpenCV-4.5.4-8-g3e4c170df4-aarch64-dev.deb   \
     && dpkg -i OpenCV-4.5.4-8-g3e4c170df4-aarch64-libs.deb  \
     && dpkg -i OpenCV-4.5.4-8-g3e4c170df4-aarch64-licenses.deb \
     && dpkg -i OpenCV-4.5.4-8-g3e4c170df4-aarch64-python.deb \
     && dpkg -i OpenCV-4.5.4-8-g3e4c170df4-aarch64-samples.deb




# Install Cmake
RUN cd /tmp && \
    wget https://github.com/Kitware/CMake/releases/download/v3.21.4/cmake-3.21.4-linux-aarch64.sh && \
    chmod +x cmake-3.21.4-linux-aarch64.sh && \
    ./cmake-3.21.4-linux-aarch64.sh --prefix=/usr/local --exclude-subdir --skip-license && \
    rm ./cmake-3.21.4-linux-aarch64.sh

# Install PyPI packages
RUN pip3 install --upgrade pip
RUN pip3 install setuptools>=41.0.0
#COPY requirements.txt /tmp/requirements.txt
#RUN pip3 install -r /tmp/requirements.txt
#RUN pip3 install jupyter jupyterlab
## Workaround to remove numpy installed with tensorflow
#RUN pip3 install --upgrade numpy

# Download NGC client
RUN cd /usr/local/bin && wget https://ngc.nvidia.com/downloads/ngccli_arm64.zip && unzip ngccli_arm64.zip && chmod u+x ngc-cli/ngc && rm ngccli_arm64.zip ngc-cli.md5 && echo "no-apikey\nascii\n" | ngc-cli/ngc config set

# Set environment and working directory
ENV TRT_LIBPATH /usr/lib/aarch64-linux-gnu/
ENV TRT_OSSPATH /workspace/TensorRT
ENV PATH="${PATH}:/usr/local/bin/ngc-cli"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${TRT_OSSPATH}/build/out:${TRT_LIBPATH}"
WORKDIR /workspace

USER trtuser
RUN ["/bin/bash"]