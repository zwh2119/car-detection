
FROM adrianosantospb/tensorrt-pytorch-opencv-arm64:latest
MAINTAINER Wenhui Zhou

ENV export LC_ALL=C.UTF-8
ENV export LANG=C.UTF-8

RUN pip3 install --upgrade pip

RUN pip3 install click==8.0.4 -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY ./requirements.txt ./
RUN pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY car_detection_trt.py service_server.py log.py ./

CMD ["uvicorn", "service_server:app", "--host=0.0.0.0", "--port=9001", "--log-level=debug", "--workers=2", "--limit-concurrency=3"]
