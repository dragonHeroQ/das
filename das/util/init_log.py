import logging
import os

_a = set()


def init_logger(logger, log_folder=None, level='INFO', name="__name__"):

    if name in _a:
        return logger
    else:
        _a.add(name)

    if level == 'INFO':
        level = logging.INFO
    elif level == 'DEBUG':
        level = logging.DEBUG
    elif level == 'ERROR':
        level = logging.ERROR
    elif level == 'CRITICAL':
        level = logging.CRITICAL

    if log_folder is None:
        log_folder = os.path.join("~", ".das", "logs")
    else:
        log_folder = log_folder
    if not os.path.exists(log_folder):
        os.makedirs(log_folder, exist_ok=True)

    logger.setLevel(level=level)

    if not os.path.exists(log_folder):
        os.chdir(log_folder)
        f = open(os.path.join(log_folder, "das_runtime.log"), 'w')
        f.close()
        os.chdir("..")

    # logging.basicConfig(datefmt='%Y-%m/%d %A %H:%M:%S')
    # init logger
    handler = logging.FileHandler(os.path.join(log_folder, "das_runtime.log"))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s %(module)s.%(funcName)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)

    return logger
