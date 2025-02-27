import jax
import jax.numpy as jnp
import numpy as np
import time
import pandas as pd
import os

# optimization
os.environ["LIBTPU_INIT_ARGS"] = "--xla_tpu_scoped_vmem_limit_kib=65536"

# D_TYPES = [jnp.bfloat16, jnp.float8_e5m2]
D_TYPES = [jnp.bfloat16]

MATRIX_DIMS = [
    (1024, 8192, 1280),
    (1024, 1024, 8192),
    (1024, 8192, 7168),
    (1024, 3584, 8192),
    (8192, 8192, 8192),
    (16384, 16384, 16384),
    (32768, 32768, 32768),
]

NUM_TRIES = 100

# Enable Tracing
jax.profiler.start_trace("/tmp/tensorboard")

@jax.jit
def matmul(a, b):
    return a @ b

result = []
for dtype in D_TYPES:
    for m, n, k in MATRIX_DIMS:
        a = jax.random.normal(jax.random.key(0), (m, n)).astype(dtype)
        b = jax.random.normal(jax.random.key(0), (n, k)).astype(dtype)

        # TODO: warm up
        jax.block_until_ready(matmul(a, b))

        time_taken = []
        for _ in range(NUM_TRIES):
            start_time = time.time()
            jax.block_until_ready(matmul(a, b))
            end_time = time.time()
            time_taken.append(end_time - start_time)
        
        time_median = np.median(time_taken)
        # MXU is done in units of bf16.
        mxu_bytes_per_op = 2
        # 2 ops (multiple and add)
        compute_flops = m * n * k * a.itemsize * 2 / mxu_bytes_per_op
        tflops_per_s = compute_flops / time_median / 1e12
        
        result.append({
            "dtype": dtype.__name__,
            "matrix_dimensions": (m, n, k),
            "time_median_secs": time_median,
            "tflops_per_sec": tflops_per_s,
        })
        print(
            f"dtype: {dtype}, matrix Dimensions: ({m}, {n}, {k}), time taken (median): {time_median} secs, TFLOPs/sec: {tflops_per_s}"
        )

jax.profiler.stop_trace()

# output result
pd.DataFrame(result).to_csv("/tmp/matmul_benchmark_result.tsv", sep="\t", index=False)