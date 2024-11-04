from loguru import logger


from functools import wraps
from time import time


def log_model_loading(model_info):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                logger.info(f'Model {model_info} is loading...')
                time_start = time()
                ret = func(*args, **kwargs)
                time_end = time()
                time_used = "{:.2f}".format(time_end - time_start)
                logger.info(f'Model {model_info} is loaded for {time_used} seconds.')
                return ret
            except AssertionError as e:
                logger.exception(e)
                logger.critical(f'Model {model_info} failed to load.')

        return wrapper

    return decorator