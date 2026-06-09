from __future__ import annotations

import os
import sys
from pprint import pformat

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cases.case_batch_utils import (
    collect_batch_log_rows,
    ensure_case_log_dir,
    export_batch_log_summary_csv,
)


def build_summary_csv_path() -> str:
    return os.path.join(ensure_case_log_dir(), "log_summary.csv")


def main() -> None:
    rows = collect_batch_log_rows()
    if not rows:
        print("No parsable batch logs found under cases/log.", flush=True)
        return

    output_file_path = export_batch_log_summary_csv(
        rows,
        output_file_path=build_summary_csv_path(),
    )

    print("=== Log Summary Ready ===", flush=True)
    print(f"rows: {len(rows)}", flush=True)
    print(f"csv: {output_file_path}", flush=True)
    print("\n=== Example Row ===", flush=True)
    print(
        pformat(
            {
                "label": rows[0].get("label"),
                "scenario_name": rows[0].get("scenario_name"),
                "user_count": rows[0].get("user_count"),
                "arrival_rate_rps": rows[0].get("arrival_rate_rps"),
                "long_context_ratio": rows[0].get("long_context_ratio"),
                "long_context_prompt_tokens": rows[0].get(
                    "long_context_prompt_tokens"
                ),
                "elapsed_seconds": rows[0].get("elapsed_seconds"),
                "simulation_total_time_ms": rows[0].get("simulation_total_time_ms"),
                "request_throughput_rps": rows[0].get("request_throughput_rps"),
                "output_token_throughput_tps": rows[0].get(
                    "output_token_throughput_tps"
                ),
                "decode_token_throughput_tps": rows[0].get(
                    "decode_token_throughput_tps"
                ),
                "prefill_token_throughput_tps": rows[0].get(
                    "prefill_token_throughput_tps"
                ),
                "request_tpot_avg_ms": rows[0].get("request_tpot_avg_ms"),
                "system_tpot_avg_ms": rows[0].get("system_tpot_avg_ms"),
                "prefill_first_token_latency_avg_ms": rows[0].get(
                    "prefill_first_token_latency_avg_ms"
                ),
                "prefill_first_token_latency_max_ms": rows[0].get(
                    "prefill_first_token_latency_max_ms"
                ),
                "short_prefill_utilization": rows[0].get(
                    "short_prefill_utilization"
                ),
                "long_prefill_utilization": rows[0].get(
                    "long_prefill_utilization"
                ),
                "decode_utilization": rows[0].get("decode_utilization"),
                "log_file_name": rows[0].get("log_file_name"),
            },
            sort_dicts=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
