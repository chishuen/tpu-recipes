import jax
import jax.numpy as jnp
import numpy as np
import time
import json
import gzip
import pathlib
import os
import pandas as pd


dtype = jnp.bfloat16
num_elements = np.array([1, 2, 4, 16, 32, 64, 128, 256, 512, 1024, 2048]) * 1024**2
num_iters = 100
log_root = "/tmp/my_trace"
output_file = "/tmp/hbm_bw_result.csv"


def extract_timing(log_dir: str) -> list[float]:
  def get_trace(log_dir: str) -> dict:
    # Navigate to folder with the latest trace dump to find `trace.json.jz`
    trace_folders = (pathlib.Path(log_dir).absolute() / "plugins" / "profile").iterdir()
    latest_trace_folder = max(trace_folders, key=os.path.getmtime)
    trace_jsons = latest_trace_folder.glob("*.trace.json.gz")
    try:
      trace_json, = trace_jsons
    except ValueError as value_error:
      raise ValueError(f"Invalid trace folder: {latest_trace_folder}") from value_error

    with gzip.open(trace_json, "rb") as f:
      trace = json.load(f)
    
    return trace
  

  def get_eligible_events(trace: dict, eligible_event: str) -> list[dict]:
    ret = []
    for e in trace["traceEvents"]:
      if "name" in e and e["name"] == eligible_event:
        ret.append(e)
    return ret

  
  trace = get_trace(log_dir)
  eligible_events = get_eligible_events(trace, "copy.1")
  return [e["dur"] / 1e6 for e in eligible_events]


result = []
for i, n in enumerate(num_elements):
  log_dir = os.path.join(log_root, f"{i}")
  a = jax.random.normal(jax.random.key(0), (n,)).astype(dtype)

  time_taken = []
  with jax.profiler.trace(log_dir):
    for _ in range(num_iters):
      b = a.copy()
      start_time = time.time()
      b.block_until_ready()
      end_time = time.time()
      time_taken.append(end_time - start_time)
  
  durations = extract_timing(log_dir)
  assert len(durations) == num_iters
  
  tensor_size = n * a.itemsize
  time_median = np.median(durations)
  bw_gbps = (tensor_size * 2) / time_median / 1e9
  result.append({
      "dtype": dtype.__name__,
      "tensor_size_bytes": n * a.itemsize,
      "time_median_ms": time_median * 1000,
      "time_mean_ms": np.mean(durations) * 1000,
      "time_min_ms": np.min(durations) * 1000,
      "time_median_time_ms": np.median(time_taken) * 1000,
      "bandwidth_gbps": bw_gbps,
  })

  print(f"Tensor size: {tensor_size / 1024**2} MBs, time taken (median): {time_median * 1000:.4f} ms, memory bandwidth: {bw_gbps:.2f} GBps")

pd.DataFrame(result).to_csv(output_file, index=False)
