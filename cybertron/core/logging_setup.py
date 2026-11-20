import logging


def setup(level):
    log = logging.getLogger("cybertron")
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        log.addHandler(handler)
    log.setLevel(level)
