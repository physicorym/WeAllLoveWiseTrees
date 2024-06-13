import asyncio
from contextlib import suppress

from aiohttp import web

from log import init_log
from server import Server


def main():
    init_log(stdout_level="DEBUG")
    server = Server()

    async def run_server_background_tasks(_app):
        task = asyncio.create_task(server.assign_task())
        yield
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    app = web.Application()
    app.cleanup_ctx.append(run_server_background_tasks)
    app.add_routes([web.post('/', server.detect)])
    app.add_routes([web.get('/', server.get_task_result)])
    app.add_routes([web.get('/task', server.get_tasks_result)])
    web.run_app(app, port=8000)


if __name__ == '__main__':
    main()
