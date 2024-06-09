import asyncio
import json
import logging
from asyncio import Queue
from dataclasses import dataclass
from typing import Union

import aiohttp
from aiohttp import web
from aiohttp.web import WebSocketResponse
from aiohttp.web_request import Request
from aiohttp.web_response import Response


logger = logging.getLogger(__name__)


@dataclass
class Task:
    task_id: int
    layout_name: str
    crop_content: bytes


class Server:
    queue: Queue
    tasks: dict[int, Union[Task, None]]

    def __init__(self):
        self.queue = asyncio.Queue()
        self.tasks = {}

    async def get_task_result(self, request: Request) -> Response:
        task_id = request.query["task_id"]
        task = await self._get_task(task_id)

        if task is None:
            return web.Response(text=f"the task with {task_id} is not found")

        return web.json_response(task)

    async def detect(self, request: Request) -> Response:
        data = await request.post()
        crop = data['file'].file.read()
        task_id = hash(crop)
        task = Task(task_id=task_id, crop_content=crop, layout_name=request.query["layout_name"])
        self.queue.put_nowait(task)

        return web.json_response({"task_id": task_id})

    async def get_tasks_result(self, request: Request) -> WebSocketResponse:
        """
        Listen to incoming ws messages and retrieve task results from them.
        """
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        async for msg in ws:
            logger.debug("Incoming message from Worker: %s", msg)
            data = json.loads(msg.data)
            self.tasks[data["task_id"]] = data["result"]

        logger.info("ws connection is closed")

        return ws

    async def assign_task(self):
        """
        Wait for the new request and assign a task to Worker.
        """

        while True:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect('http://127.0.0.1:8888/') as ws:
                    new_task: Task = await self.queue.get()
                    self.tasks[new_task.task_id] = None
                    logger.info("Sending new task: %s", new_task)
                    await ws.send_json(new_task)

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
