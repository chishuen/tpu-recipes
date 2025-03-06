import jax
import jax.numpy as jnp
import os
import pandas as pd
import re

from benchmark_utils import run_bench

# optimization
os.environ["LIBTPU_INIT_ARGS"] = "--xla_tpu_scoped_vmem_limit_kib=65536"

dtypes = [jnp.bfloat16]
mat_dims = [
    (1024, 8192, 1280),
    (1024, 1024, 8192),
    (1024, 8192, 7168),
    (1024, 3584, 8192),
    (8192, 8192, 8192),
    (1536, 1024, 24576),
    (1024, 24576, 1536),
    (16384, 24576, 5120),
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (16384, 16384, 16384),
]
num_iter = 100

def matmul(a, b):
    return a @ b

results = []
for dtype in dtypes:
  for m, n, k in mat_dims:
    a = jax.random.normal(jax.random.key(0), (m, n)).astype(dtype)
    b = jax.random.normal(jax.random.key(0), (n, k)).astype(dtype)

    compiled = jax.jit(matmul).lower(a, b).compile()
    result = run_bench(lambda: jax.block_until_ready(compiled(a, b)), num_iter=num_iter, log_dir="/tmp/matmul", event_matcher=re.compile(r"jit_matmul.*"))

    # MXU is done in units of bf16.
    mxu_bytes_per_op = 2
    # 2 ops (multiple and add)
    compute_flops = m * n * k * a.itemsize * 2 / mxu_bytes_per_op

    results.append({
        "dtype": dtype.__name__,
        "matrix_dimensions": (m, n, k),
        "time_median_ms": result.time_median * 1e3,
        "tflops_per_sec_median": compute_flops / result.time_median / 1e12,
        "time_min_ms": result.time_min * 1e3,
        "tflops_per_sec_max": compute_flops / result.time_min / 1e12,
    })
    print(f"dtype: {dtype.__name__}, matrix Dimensions: ({m}, {n}, {k}), time taken (median): {result.time_median * 1e3} ms, TFLOPs/sec: {compute_flops / result.time_median / 1e12}")

# output result
pd.DataFrame(results).to_csv("/tmp/matmul_result.tsv", sep="\t", index=False)