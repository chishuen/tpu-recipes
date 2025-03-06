import jax
import jax.numpy as jnp
import numpy as np
import os
import pandas as pd
import re

from benchmark_utils import run_bench


dtype = jnp.bfloat16
num_elements = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]) * 1024**2
num_iter = 100
log_root = "/tmp/my_trace"
output_file = "/tmp/hbm_bw_result.csv"

def my_copy(a):
  return a.copy()

results = []
for n in num_elements:
  a = jax.random.normal(jax.random.key(0), (n,)).astype(dtype)
  compiled = jax.jit(my_copy).lower(a).compile()
  result = run_bench(lambda: jax.block_until_ready(compiled(a)), num_iter=num_iter, log_dir="/tmp/hbm", event_matcher=re.compile(r"jit_my_copy.*"))
  tensor_size = n * a.itemsize
  bw_gbps = (tensor_size * 2) / result.time_median / 1e9  # read + write = 2
  results.append({
      "dtype": dtype.__name__,
      "tensor_size_bytes": n * a.itemsize,
      "time_median_ms": result.time_median * 1000,
      "time_min_ms": result.time_min * 1000,
      "bandwidth_gbps_median": bw_gbps,
      "bandwidth_gbps_max": (tensor_size * 2) / result.time_min / 1e9,
  })

  print(f"Tensor size: {tensor_size / 1024**2} MBs, time taken (median): {result.time_median * 1000:.4f} ms, bandwidth: {bw_gbps:.2f} GBps")

pd.DataFrame(results).to_csv(output_file, index=False)
