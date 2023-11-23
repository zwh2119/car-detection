import time
import contextlib
import threading
import asyncio

import base64

from fastapi import FastAPI
from fastapi.routing import APIRoute
from starlette.responses import JSONResponse
from starlette.requests import Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from car_detection import CarDetection
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
            'device': 'cuda:0'
        })

        self.app.add_middleware(
            CORSMiddleware, allow_origins=["*"], allow_credentials=True,
            allow_methods=["*"], allow_headers=["*"],
        )

    async def cal(self, request: Request):
        data = await request.json()
        content = base64.b64decode(data['input'])

        result = await self.estimator(content)

        assert type(result) is dict

        return result


app_server = ServiceServer()
app = app_server.app
