ARG TENSORRT="8.2"
ARG CUDA="11.2"

FROM hakuyyf/tensorrtx:trt${TENSORRT}_cuda${CUDA}
MAINTAINER Wenhui Zhou

COPY ./requirements.txt ./
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY car_detection_trt.py service_server.py log.py ./

CMD ["uvicorn", "service_server:app", "--host=0.0.0.0", "--port=9001", "--log-level=debug", "--workers=2", "--limit-concurrency=3"]
