
FROM louiswe/tensorrt-yolo:latest
MAINTAINER Wenhui Zhou

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN apt-get update && apt-get install -y gnupg

RUN pip3 install --upgrade pip

#RUN pip3 install tensorrt_libs==9.0.0.post11.dev1
#RUN pip3 install tensorrt

COPY ./requirements.txt ./
RUN pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY car_detection_trt.py service_server.py log.py ./

CMD ["uvicorn", "service_server:app", "--host=0.0.0.0", "--port=9001", "--log-level=debug", "--workers=2", "--limit-concurrency=3"]
