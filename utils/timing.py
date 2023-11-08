from contextlib import contextmanager
import timeit
from datetime import datetime

@contextmanager
def execution_timing(label: str):
    start = timeit.default_timer()
    start_time = datetime.now()
    print(f"{label}: started at {start_time.strftime('%H:%M:%S')}")
    try:
        yield
    finally:
        end = timeit.default_timer()
        print(f"{label}: finished at {datetime.now().strftime('%H:%M:%S')} and took {round((end - start)*1000)}ms")

