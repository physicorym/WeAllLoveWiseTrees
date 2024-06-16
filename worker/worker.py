import asyncio
import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import Pool
from typing import Union

import numpy as np
from aiohttp import web
from aiohttp.web_request import Request
from aiohttp.web_response import Response

from detection.base import detect


logger = logging.getLogger(__name__)


@dataclass
class Task:
    task_id: int
    layout_name: str
    crop_content: bytes
    crop_height: int
    crop_width: int


class Worker:
    task_queue: asyncio.Queue
    tasks: dict[int, Union[Task, None]]

    def __init__(self):
        self.task_queue = asyncio.Queue()
        self.tasks = {}

    async def detect(self, request: Request) -> Response:
        logger.info("Incoming a new request: %s", id(request))
        data = await request.post()
        crop = data['file'].file.read()
        task_id = random.randint(0, hash(time.time()))
        logger.info("Task id: %s", task_id)
        task = Task(
            task_id=task_id,
            crop_content=crop,
            layout_name=request.query["layout_name"],
            crop_height=int(request.query["crop_height"]),
            crop_width=int(request.query["crop_width"]),
        )
        self.task_queue.put_nowait(task)

        return web.json_response({"task_id": task_id})

    async def get_task_result(self, request: Request) -> Response:
        task_id = int(request.query["task_id"])
        task = await self._get_task(task_id)

        if task is None:
            return web.Response(text=f"the task with {task_id} is not found")

        return web.json_response(task)

    async def process_task(self):
        with Pool(processes=1) as pool:
            while True:
                try:
                    while self.task_queue.empty():
                        await asyncio.sleep(0)
                    task = self.task_queue.get_nowait()
                    logger.info("Got new task: %s", task.task_id)
                    crop = np.frombuffer(task.crop_content)
                    result = pool.apply_async(self.detect_with_measuring,
                                              (task.layout_name, crop, task.crop_height, task.crop_width)).get()
                    result["layout_name"] = task.layout_name
                    self.tasks[task.task_id] = result
                    logger.info("Task %s is processed", task.task_id)
                    await asyncio.sleep(0)
                except Exception as exc:
                    logger.error("An exception occurred: %s %s", type(exc), exc)

    @staticmethod
    def detect_with_measuring(layout_name: str, crop: np.ndarray, height: int, width: int) -> dict:
        logger.info("Start detection")
        start_perf_counter = time.perf_counter()
        start = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        result = detect(layout_name, crop, height=height, width=width)
        end = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        logger.info("Detection got %s seconds", time.perf_counter() - start_perf_counter)

        result["start"] = start
        result["end"] = end

        return result

    async def _get_task(self, task_id) -> Union[Task, None]:
        """
        Get the task by id.

        If the task is still in progress, waits for it.
        Returns None if <task_id> is not found in the dictionary.
        """

        if task_id not in self.tasks:
            return None

        while not self.tasks[task_id]:
            await asyncio.sleep(0)

        return self.tasks.pop(task_id)
