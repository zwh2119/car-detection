
FROM yuefan2022/tensorrt-ubuntu20.04-cuda11.6
MAINTAINER Wenhui Zhou

COPY ./requirements.txt ./
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# TODO：libmyplugins.so yolov5s.engine will be removed in later version
COPY car_detection_trt.py service_server.py log.py libmyplugins.so yolov5s.engine ./

CMD ["uvicorn", "service_server:app", "--host=0.0.0.0", "--port=9001", "--log-level=debug", "--workers=2", "--limit-concurrency=3"]
