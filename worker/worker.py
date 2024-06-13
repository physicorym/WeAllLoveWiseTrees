import asyncio
import json
import logging
import time
from datetime import datetime
from multiprocessing import Pool
from typing import Optional

import aiohttp
import numpy as np
from aiohttp import web
from aiohttp.web import WebSocketResponse
from aiohttp.web_request import Request

from detection import detect


logger = logging.getLogger(__name__)


class Worker:
    awaiting_tasks_queue: asyncio.Queue
    ready_tasks_queue: asyncio.Queue

    def __init__(self):
        self.awaiting_tasks_queue = asyncio.Queue()
        self.ready_tasks_queue = asyncio.Queue()

    async def receive_tasks(self, request: Request) -> WebSocketResponse:
        """
        Listen to incoming ws messages, receive tasks and process them.
        """
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        async for msg in ws:
            data = json.loads(msg.data)
            logger.debug("New task from Server: %s", data["task_id"])
            self.awaiting_tasks_queue.put_nowait(data)
            await asyncio.sleep(0)

        logger.info("ws connection is closed")

        return ws

    async def process_task(self):
        with Pool(processes=1) as pool:
            while True:
                try:
                    while self.awaiting_tasks_queue.empty():
                        await asyncio.sleep(0)
                    data = self.awaiting_tasks_queue.get_nowait()
                    logger.info("Got new task: %s", data["task_id"])
                    layout_name = data["layout_name"]
                    crop = np.frombuffer(data["crop_content"].encode())
                    result = pool.apply_async(self.detect_with_measuring, (layout_name, crop))
                    result = result.get()
                    result["layout_name"] = data["layout_name"]
                    result_json = json.dumps({"task_id": data["task_id"], "result": result})
                    self.ready_tasks_queue.put_nowait(result_json)
                    logger.info("Task %s is processed", data["task_id"])
                    await asyncio.sleep(0)
                except Exception as exc:
                    logger.error("An exception occurred: %s %s", type(exc), exc)

    async def send_task_results(self):
        ready_task: Optional[str] = None
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect('http://127.0.0.1:8000/task') as ws:
                        while True:
                            while self.ready_tasks_queue.empty():
                                await asyncio.sleep(0)
                            ready_task = self.ready_tasks_queue.get_nowait()
                            await ws.send_json(ready_task)
            except Exception as exc:
                logger.error("An exception occurred: %s", exc)
                if ready_task:
                    self.ready_tasks_queue.put_nowait(ready_task)
                    ready_task = None
                await asyncio.sleep(0)

    @staticmethod
    def detect_with_measuring(layout_name: str, crop: np.ndarray) -> dict:
        logger.info("Start detection")
        start_perf_counter = time.perf_counter()
        start = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        result = detect(layout_name, crop)
        end = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        logger.info("Detection got %s seconds", time.perf_counter() - start_perf_counter)

        result["start"] = start
        result["end"] = end

        return result
