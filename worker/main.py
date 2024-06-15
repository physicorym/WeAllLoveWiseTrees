import asyncio
import os
from contextlib import suppress

from aiohttp import web

from log import init_log
from worker import Worker


def main():
    app_port = int(os.getenv("APP_PORT", "8000"))

    init_log(stdout_level="DEBUG")
    worker = Worker()
    
    async def run_worker_background_tasks(_app):
        tasks = [
            asyncio.create_task(worker.process_task()),
        ]
        yield

        for task in tasks:
            task.cancel()
        for task in tasks:
            with suppress(asyncio.CancelledError):
                await task

    app = web.Application()
    app.cleanup_ctx.append(run_worker_background_tasks)
    app.add_routes([web.post('/', worker.detect)])
    app.add_routes([web.get('/', worker.get_task_result)])
    web.run_app(app, port=app_port)


if __name__ == '__main__':
    main()
