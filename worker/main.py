import asyncio
from contextlib import suppress

from aiohttp import web

from log import init_log
from worker import Worker


def main():
    init_log(stdout_level="DEBUG")
    worker = Worker()
    
    async def run_worker_background_tasks(_app):
        tasks = [
            asyncio.create_task(worker.process_task()),
            asyncio.create_task(worker.send_task_results())
        ]
        yield

        for task in tasks:
            task.cancel()
        for task in tasks:
            with suppress(asyncio.CancelledError):
                await task

    app = web.Application()
    app.cleanup_ctx.append(run_worker_background_tasks)
    app.add_routes([web.get('/', worker.receive_tasks)])
    web.run_app(app, port=8888)


if __name__ == '__main__':
    main()
