import logging
import sys


def init_log(stdout_level: str = 'INFO'):
    """Initialize root logger"""
    if hasattr(init_log, '_called'):
        logging.warning("init_log must be called only once")
        return

    init_log._called = True
    logger = logging.getLogger()

    # Remove default handler
    if logger.handlers:
        logger.handlers = []

    fmt: str = "%(levelname)8s %(asctime)s <%(module)s %(lineno)s> %(message)s"
    date_fmt: str = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=date_fmt)

    h_stdout = logging.StreamHandler(sys.stdout)
    h_stdout.setLevel(stdout_level)
    h_stdout.addFilter(lambda record: record.levelno <= logging.WARNING)
    h_stdout.setFormatter(formatter)
    logger.addHandler(h_stdout)

    h_stderr = logging.StreamHandler(sys.stderr)
    h_stderr.setLevel(logging.ERROR)
    h_stderr.setFormatter(formatter)
    logger.addHandler(h_stderr)

    logger.setLevel(min(h_stdout.level, h_stderr.level))
