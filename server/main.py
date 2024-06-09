from aiohttp import web

from log import init_log
from server import Server


def main():
    init_log(stdout_level="INFO")
    server = Server()

    app = web.Application()
    app.add_routes([web.post('/', server.detect)])
    app.add_routes([web.get('/', server.get_task_result)])
    app.add_routes([web.get('/task', server.get_tasks_result)])
    web.run_app(app, port=8000)


if __name__ == '__main__':
    main()
