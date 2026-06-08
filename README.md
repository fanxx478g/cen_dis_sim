# cen_dis_sim

`cen_dis_sim` is a discrete-event simulator for large-model inference clusters.

It currently models:
- Request generation by session, including first-turn arrivals and follow-up turns.
- Prompt-length buckets for short-context and long-context traffic.
- Separate `prefill -> decode` lifecycle stages.
- Resource hierarchy `cluster -> pool -> instance`.
- Independent batching limits for `prefill` and `decode`.
- Continuous batching semantics for decode resident sets.
- Request-, pool-, and system-level metrics for latency, batching, queueing, TPOT, and throughput.

## Project Layout

- [`cases`](/C:/Project/cen_dis_sim/cases): runnable demos and scaling benchmark scripts.
- [`simulator`](/C:/Project/cen_dis_sim/simulator): simulator core, config models, scheduling, and metrics.
- [`test`](/C:/Project/cen_dis_sim/test): regression tests for routing, batching, request generation, and metric semantics.

## Run

Run the demo:

```powershell
python .\cases\full_config_demo.py
```

Run tests:

```powershell
python -m unittest discover .\test
```

Run scaling benchmark:

```powershell
python .\cases\benchmark_scaling.py --suite requests --repeat 1
```

## Request Generation

Key request-generation knobs:
- `user_count`: number of independent users or sessions.
- `min_turns_per_user` / `max_turns_per_user`: inclusive turn-count range per session.
- `new_user_arrival_mean_seconds`: mean inter-arrival time for first-turn requests.
- `followup_arrival_mean_seconds` / `followup_arrival_std_seconds`: follow-up delay distribution after the previous turn completes.
- `accumulate_context_across_turns`: when `True`, later turns reuse accumulated history and outputs.
- `followup_prompt_tokens`: fresh user-input tokens appended on follow-up turns.
- `short_context_probability`: probability that a first turn is sampled from the short-context bucket.
- `initial_prompt_variation_ratio`: prompt jitter around the bucket center.
- `short_context_prompt_tokens` / `long_context_prompt_tokens`: bucket-center prompt lengths.
- `short_context_output_tokens_min/max` and `long_context_output_tokens_min/max`: output-length ranges per bucket.

When `accumulate_context_across_turns=True`, each later turn uses:

```text
prompt_tokens = history_tokens + new_prompt_tokens
history_tokens += new_prompt_tokens + output_tokens
```

When `accumulate_context_across_turns=False`, each turn falls back to the first-turn prompt size.

## Metric Semantics

### Latency

- `prefill_first_token_latency_p50_ms` / `p95_ms`
  - Preserves the simulator's historical TTFT semantics.
  - This is the current simulator assumption: first token is produced when `prefill` finishes.
  - It is not the same as real-system end-to-end TTFT.
- `e2e_p50_ms` / `p95_ms`
  - End-to-end request latency from arrival to final token completion.
- `prefill_queue_p50_ms` / `p95_ms`
  - Request-level time spent waiting for prefill admission.
- `decode_queue_p50_ms` / `p95_ms`
  - Request-level cumulative time spent waiting for decode admission across decode steps.

### TPOT

- `request_tpot_ms`
  - Request-level TPOT:

```text
request_tpot_ms = (finish_time_ms - first_token_time_ms) / max(output_tokens - 1, 1)
```

- `request_tpot_p50_ms` / `p95_ms`
  - Distribution of request-level TPOT across completed requests.
- `system_tpot_avg_ms`
  - Average of all completed requests' `request_tpot_ms`.

### Throughput

All throughput metrics use the external reporting window `active_window`:

```text
active_window_ms = last_finish_time_ms - first_arrival_time_ms
```

Reported throughput metrics:
- `request_throughput_rps`
  - `completed_requests / active_window_s`
- `output_token_throughput_tps`
  - `total_output_tokens / active_window_s`
- `decode_token_throughput_tps`
  - `total_decode_tokens / active_window_s`
- `prefill_token_throughput_tps`
  - `total_prompt_tokens / active_window_s`

Notes:
- `total_output_tokens` counts the full response length.
- `total_decode_tokens` counts `max(output_tokens - 1, 0)` per request because the simulator currently assumes the first token is produced by `prefill`.
- `total_prompt_tokens` counts full prompt lengths admitted to prefill.

### SLA-Style Counters

- `requests_with_tpot_le_50ms`
- `requests_with_tpot_le_50ms_ratio`
- `requests_with_prefill_first_token_latency_le_2s`
- `requests_with_prefill_first_token_latency_le_2s_ratio`

These help evaluate how much traffic stays within target latency thresholds.

## Benchmark Output

`cases/benchmark_scaling.py` now separates simulator runtime from simulated service metrics:
- `sim_runtime_ms_*`: Python simulator execution time, for tooling cost only.
- `request_tput_rps`: simulated request throughput.
- `output_tput_tps`: simulated output-token throughput.
- `prefill_first_token_p95_ms`: historical TTFT-style latency.
- `request_tpot_avg_ms`: average request-level TPOT.
- `decode_q95_ms`: decode queue tail latency.

## Current Model Limitations

- Decode duration is still a placeholder linear model, not real profiling data.
- The simulator still treats `prefill` completion as first-token production.
- Cross-cluster migration cost after the first decode step is not fully reflected in latency metrics.

## Next Improvements

- Replace decode timing with profiling-backed latency curves.
- Add real-system TTFT components such as first decode step and network return time.
- Add richer routing policies and compare them under the new throughput and TPOT metrics.
