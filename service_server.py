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


class Server(uvicorn.Server):
    def install_signal_handlers(self):
        pass

    @contextlib.contextmanager
    def run_in_thread(self):
        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()
        try:
            yield thread
        finally:
            self.should_exit = True
            thread.join()


class BaseServer:
    DEBUG = True
    WAIT_TIME = 15

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
        # print(data)

        print(time.time(), f'{data["id"]} start')

        result = await self.estimator(field_codec_utils.decode_image(data['image']))

        print(time.time(), f'{data["id"]} end  result:{result}')

        return {'id': data['id'], 'result': result}

    def wait_stop(self, current):
        """wait the stop flag to shutdown the server"""
        while 1:
            time.sleep(self.WAIT_TIME)
            if not current.is_alive():
                return
            if getattr(self.app, "shutdown", False):
                return


app_server = BaseServer()
app = app_server.app

