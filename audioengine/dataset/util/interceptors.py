from functools import wraps
import time
from math import ceil

def time_logger(logger, **kwargs):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **inner_kwargs):
            startTime = time.time()
            name = kwargs.get("name", func.__name__)
            header = kwargs.get("header", None)
            footer = kwargs.get("footer", None)

            if header:
                header_msg = pad_str(header, **kwargs)
                logger.debug(header_msg, extra={'name_override': func.__name__})

            result = func(*args, **inner_kwargs)

            totalTime = time.time() - startTime

            logger.debug(f"[{name}] Elapsed Time: {totalTime}s", extra={'name_override': func.__name__})

            if footer:
                footer_msg = pad_str(footer, **kwargs)
                logger.debug(footer_msg, extra={'name_override': func.__name__})
            return result

        return wrapper

    return decorator

def pad_str(msg, **kwargs):
    length = kwargs.get("padding_length", 32)
    symbol = kwargs.get("padding_symbol", "*")

    symbol_count = ceil((length - len(msg)) / 2) - 1

    symbols = symbol * symbol_count
    return f"{symbols} {msg} {symbols}"
