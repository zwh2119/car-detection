FROM nvidia/cuda:11.7.1-runtime-ubuntu20.04

MAINTAINER Wenhui Zhou

#添加python的安装包
ADD Python-3.10.9.tar.xz /opt

ENV DEBIAN_FRONTEND=noninteractive

RUN mkdir /usr/local/python-3.10
#安装依赖

RUN apt-get update && apt-get install gcc -y && apt-get install make -y \
		&& apt-get install vim -y && apt-get install openssl -y \
		&& apt-get install libssl-dev -y && apt-get install python3-pip -y
RUN /opt/Python-3.10.9/configure --prefix=/usr/local/python-3.10 \
		&& make && make install

COPY . .

CMD ["gunicorn", "service_server:app", "-c", "./gunicorn.conf.py"]


