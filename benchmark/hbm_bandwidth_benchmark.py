import jax
import jax.numpy as jnp
import numpy as np
import time

num_elements = 1 * 1024**3
num_tries = 100

a = jax.random.normal(jax.random.key(0), (num_elements,)).astype(jnp.float32)
tensor_size = num_elements * a.itemsize
time_taken = []

jax.profiler.start_trace("/tmp/tensorboard_bw")

for _ in range(num_tries):
  b = a.copy()
  start_time = time.time()
  b.block_until_ready()
  end_time = time.time()
  time_taken.append(end_time - start_time)

jax.profiler.stop_trace()

time_median = np.median(time_taken)
bw_gbps = (tensor_size * 2) / (time_median) / 1e9

print(f"Tensor size: {tensor_size / 1024**3} GBs, time taken (median): {time_median*1000} ms, memory bandwidth: {bw_gbps:.2f} GB/s")
