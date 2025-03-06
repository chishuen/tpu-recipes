"""Utilities for benchmarks."""

import gzip
import json
import jax
import numpy as np
import os
import pathlib
import re

from dataclasses import dataclass
from typing import Callable

@dataclass
class BenchmarkResult:
  time_median: float = 0
  time_min: float = 0


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


def get_eligible_events(trace: dict, event_matcher: re.Pattern = None) -> list[dict]:
  if not "traceEvents" in trace:
    raise KeyError(f"Key 'traceEvents' not found in trace.")

  ret = []
  for e in trace["traceEvents"]:
    if "name" in e and event_matcher.match(e["name"]):
      ret.append(e)
  return ret


def get_benchmark_result(events: list[dict]):
  try:
    durations = [e["dur"] / 1e6 for e in events]
  except KeyError as ker_error:
    raise ker_error
  
  return BenchmarkResult(
    time_median=np.median(durations),
    time_min=np.min(durations),
  )


def run_bench(fn: Callable, num_iter=1, warmup_iter=0, log_dir="/tmp", event_label="my_func", event_matcher: re.Pattern = None):
  # warm up
  for _ in range(warmup_iter):
    fn()

  with jax.profiler.trace(log_dir):
    for _ in range(num_iter):
      jax.clear_caches()
      with jax.profiler.TraceAnnotation(event_label):
        fn()
  
  trace = get_trace(log_dir)

  if not event_matcher:
    event_matcher = re.compile(event_label)
  events = get_eligible_events(trace, event_matcher)
  assert len(events) == num_iter

  return get_benchmark_result(events)