FROM  nvcr.io/nvidia/l4t-tensorflow:r32.4.3-tf1.15-py3

ENV DEBIAN_FRONTEND=noninteractive

COPY car_detection/lib /pdk_files

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

RUN apt-get --purge remove  cuda*

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


RUN dpkg -i /pdk_files/cuda-repo-l4t-10-2-local_10.2.460-1_arm64.deb
RUN apt-key add /var/cuda-repo*/*.pub
RUN apt-get -y update
RUN apt-get -f -y install
RUN mv /var/lib/dpkg/info /var/lib/dpkg/info.bak && mkdir /var/lib/dpkg/info
#RUN apt-get -y upgrade
RUN apt-get -y update

RUN apt-get -y update
RUN  dpkg -r --force-all cuda-misc-headers-10-2
RUN apt-get -y install cuda-toolkit-10-2

RUN     dpkg -i /pdk_files/libcudnn8_8.2.1.32-1+cuda10.2_arm64.deb \
     && dpkg -i /pdk_files/libcudnn8-dev_8.2.1.32-1+cuda10.2_arm64.deb \
     && dpkg -i /pdk_files/libcudnn8-samples_8.2.1.32-1+cuda10.2_arm64.deb \
     && dpkg -i /pdk_files/libnvinfer8_8.2.1-1+cuda10.2_arm64.deb  \
     && dpkg -i /pdk_files/libnvinfer-dev_8.2.1-1+cuda10.2_arm64.deb \
     && dpkg -i /pdk_files/libnvparsers8_8.2.1-1+cuda10.2_arm64.deb   \
     && dpkg -i /pdk_files/libnvparsers-dev_8.2.1-1+cuda10.2_arm64.deb \
     && dpkg -i /pdk_files/libnvinfer-plugin8_8.2.1-1+cuda10.2_arm64.deb \
     && dpkg -i /pdk_files/libnvinfer-plugin-dev_8.2.1-1+cuda10.2_arm64.deb \
     && dpkg -i /pdk_files/libnvonnxparsers8_8.2.1-1+cuda10.2_arm64.deb \
     && dpkg -i /pdk_files/libnvonnxparsers-dev_8.2.1-1+cuda10.2_arm64.deb \
     && dpkg -i /pdk_files/libnvinfer-bin_8.2.1-1+cuda10.2_arm64.deb \
     && dpkg -i /pdk_files/libnvinfer-samples_8.2.1-1+cuda10.2_all.deb  \
     && dpkg -i /pdk_files/libnvinfer-doc_8.2.1-1+cuda10.2_all.deb \
     && dpkg -i /pdk_files/graphsurgeon-tf_8.2.1-1+cuda10.2_arm64.deb \
     && dpkg -i /pdk_files/uff-converter-tf_8.2.1-1+cuda10.2_arm64.deb \
     && dpkg -i /pdk_files/python3-libnvinfer_8.2.1-1+cuda10.2_arm64.deb \
     && dpkg -i /pdk_files/python3-libnvinfer-dev_8.2.1-1+cuda10.2_arm64.deb \
     && dpkg -i /pdk_files/tensorrt_8.2.1.9-1+cuda10.2_arm64.deb \
     && dpkg -i /pdk_files/OpenCV-4.1.1-2-gd5a58aa75-aarch64-dev.deb \
     && dpkg -i /pdk_files/OpenCV-4.1.1-2-gd5a58aa75-aarch64-libs.deb \
     && dpkg -i /pdk_files/OpenCV-4.1.1-2-gd5a58aa75-aarch64-licenses.deb \
     && dpkg -i /pdk_files/OpenCV-4.1.1-2-gd5a58aa75-aarch64-python.deb \
     && dpkg -i /pdk_files/OpenCV-4.1.1-2-gd5a58aa75-aarch64-samples.deb  \
     && dpkg -i /pdk_files/OpenCV-4.5.4-8-g3e4c170df4-aarch64-dev.deb   \
     && dpkg -i /pdk_files/OpenCV-4.5.4-8-g3e4c170df4-aarch64-libs.deb  \
     && dpkg -i /pdk_files/OpenCV-4.5.4-8-g3e4c170df4-aarch64-licenses.deb \
     && dpkg -i /pdk_files/OpenCV-4.5.4-8-g3e4c170df4-aarch64-python.deb \
     && dpkg -i /pdk_files/OpenCV-4.5.4-8-g3e4c170df4-aarch64-samples.deb



RUN ["/bin/bash"]