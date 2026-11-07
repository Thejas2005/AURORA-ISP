import time, tracemalloc, numpy as np

def time_fn(fn, *args, warmup=1, runs=5):
    """(cold_ms, warm_median_ms, peak_python_heap_MB). Heap peak only; report runtime memory separately."""
    t0 = time.perf_counter(); fn(*args); cold = (time.perf_counter() - t0) * 1e3
    for _ in range(warmup): fn(*args)
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter(); fn(*args); ts.append((time.perf_counter() - t0) * 1e3)
    tracemalloc.start(); fn(*args); _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    return cold, float(np.median(ts)), peak / 1e6
