import time
import contextlib
import threading
import asyncio

from fastapi import FastAPI
from fastapi.routing import APIRoute
from starlette.responses import JSONResponse
from starlette.requests import Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from car_detection.car_detection import CarDetection
import field_codec_utils


class ServiceServer:

    def __init__(self):
        self.app = FastAPI(routes=[
            APIRoute('/predict',
                     self.cal,
                     response_class=JSONResponse,
                     methods=['POST']

                     ),
        ], log_level='trace', timeout=6000)

        self.estimator = CarDetection({
            'weights': 'yolov5s.pt',
            # 'device': 'cpu'
            'device': 'cuda:1'
        })

        self.app.add_middleware(
            CORSMiddleware, allow_origins=["*"], allow_credentials=True,
            allow_methods=["*"], allow_headers=["*"],
        )

    async def cal(self, request: Request):
        data = await request.json()

        print(time.time(), f'{data["id"]} start')

        result = await self.estimator(field_codec_utils.decode_image(data['image']))

        print(time.time(), f'{data["id"]} end  result:{result}')

        return {'id': data['id'], 'result': result}


app_server = ServiceServer()
app = app_server.app
