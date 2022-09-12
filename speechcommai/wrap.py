import functools
import time
import os

def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer

def mkdir(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_dirmaker(dir, *args, **kwargs):
        if not os.path.exists(dir):
            os.mkdir(dir)
        value = func(dir, *args, **kwargs)
        return value
    return wrapper_dirmaker