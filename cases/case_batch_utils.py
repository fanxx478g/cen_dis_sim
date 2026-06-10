from __future__ import annotations

import ast
import csv
import os
import time
import traceback
from dataclasses import is_dataclass
from pprint import pformat
from typing import Any

from simulator import LoggingConfig, SimulationConfig, SimulationEngine


def apply_overrides(target: Any, overrides: dict[str, object] | None) -> Any:
    if overrides is None:
        return target
    if not is_dataclass(target):
        raise TypeError("apply_overrides target must be a dataclass instance.")
    for field_name, value in overrides.items():
        if not hasattr(target, field_name):
            raise AttributeError(
                f"{target.__class__.__name__} has no field '{field_name}'."
            )
        setattr(target, field_name, value)
    return target


def ensure_case_log_dir() -> str:
    log_dir = os.path.join(os.path.dirname(__file__), "log")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def sanitize_tag(value: object) -> str:
    text = str(value)
    safe_chars = []
    for char in text:
        if char.isalnum():
            safe_chars.append(char)
        elif char in {".", "-"}:
            safe_chars.append("_")
    sanitized = "".join(safe_chars).strip("_")
    return sanitized or "value"


def abbreviate_scenario_label(scenario_label: str) -> str:
    mapping = {
        "distributed_2_clusters": "d2",
        "centralized_long_prefill_2_clusters": "c2",
        "distributed_5_clusters_dis_first": "d5_df",
        "centralized_short_prefill_5_clusters_dis_first": "c5_df",
    }
    return mapping.get(scenario_label, sanitize_tag(scenario_label))


def format_compact_ratio(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text.replace(".", "")


def format_prompt_tokens(value: int) -> str:
    if value % 1024 == 0:
        return f"{value // 1024}k"
    return sanitize_tag(value)


def build_batch_log_file_path(
    scenario_label: str,
    *,
    user_count: int,
    arrival_rate_rps: float,
    long_context_ratio: float,
    long_context_prompt_tokens: int,
    seed: int,
) -> str:
    log_dir = ensure_case_log_dir()
    file_name = "_".join(
        [
            abbreviate_scenario_label(scenario_label),
            f"u{sanitize_tag(user_count)}",
            f"r{sanitize_tag(arrival_rate_rps)}",
            f"lr{format_compact_ratio(long_context_ratio)}",
            f"lp{format_prompt_tokens(long_context_prompt_tokens)}",
            f"s{sanitize_tag(seed)}",
        ]
    )
    return os.path.join(log_dir, f"{file_name}.log")


def write_log_preamble(
    log_file_path: str,
    *,
    run_label: str,
    metadata: dict[str, object],
) -> None:
    with open(log_file_path, "w", encoding="utf-8") as handle:
        handle.write("=== Run Metadata ===\n")
        handle.write(f"label: {run_label}\n")
        for key, value in metadata.items():
            handle.write(f"{key}: {value}\n")
        handle.write("\n")


def append_summary_to_log(
    log_file_path: str,
    *,
    summary: dict[str, object],
    elapsed_seconds: float,
) -> None:
    with open(log_file_path, "a", encoding="utf-8") as handle:
        handle.write("\n=== Global Summary ===\n")
        handle.write(pformat(summary, sort_dicts=False))
        handle.write("\n")
        handle.write(f"===仿真时间=== {elapsed_seconds}s\n")


def append_exception_to_log(log_file_path: str, exc: BaseException) -> None:
    with open(log_file_path, "a", encoding="utf-8") as handle:
        handle.write("\n=== Run Failed ===\n")
        handle.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))


def prepare_logging_config(
    base_logging: LoggingConfig,
    *,
    log_file_path: str,
    logger_name: str,
) -> LoggingConfig:
    apply_overrides(
        base_logging,
        {
            "level": base_logging.level,
            "log_to_console": True,
            "log_to_file": True,
            "log_file_path": log_file_path,
            "logger_name": logger_name,
        },
    )
    return base_logging


def run_config_with_file_logging(
    config: SimulationConfig,
    *,
    log_file_path: str,
    run_label: str,
    metadata: dict[str, object],
    seed: int,
) -> dict[str, object]:
    write_log_preamble(log_file_path, run_label=run_label, metadata=metadata)

    started = time.time()
    try:
        engine = SimulationEngine(config=config, seed=seed)
        metrics = engine.run()
        summary = metrics.summary()
        summary.update(
            metrics.global_summary(
                pools=engine.pools,
                total_time_ms=engine.current_time_ms,
            )
        )
        elapsed_seconds = time.time() - started
        append_summary_to_log(
            log_file_path,
            summary=summary,
            elapsed_seconds=elapsed_seconds,
        )
        return {
            "summary": summary,
            "elapsed_seconds": elapsed_seconds,
            "log_file_path": log_file_path,
        }
    except BaseException as exc:
        append_exception_to_log(log_file_path, exc)
        raise


def _parse_scalar(text: str) -> object:
    text = text.strip()
    if text in {"True", "False"}:
        return text == "True"
    if text == "None":
        return None
    try:
        if any(char in text for char in [".", "e", "E"]):
            return float(text)
        return int(text)
    except ValueError:
        return text


def _parse_global_summary_log_line(content: str) -> dict[str, object]:
    parsed: dict[str, object] = {}
    summary_line: str | None = None
    for line in content.splitlines():
        if "global_summary " in line:
            summary_line = line.strip()
    if summary_line is None:
        return parsed

    payload = summary_line.split("global_summary ", 1)[1].strip()
    for item in payload.split(","):
        part = item.strip()
        if not part or "=" not in part:
            continue
        key, value = part.split("=", 1)
        parsed[key.strip()] = _parse_scalar(value)
    return parsed


def parse_batch_log(log_file_path: str) -> dict[str, object] | None:
    with open(log_file_path, "r", encoding="utf-8") as handle:
        content = handle.read()

    if "=== Run Metadata ===" not in content:
        return None
    if "=== Global Summary ===" not in content:
        return None
    if "===仿真时间===" not in content:
        return None

    metadata_start = content.index("=== Run Metadata ===") + len("=== Run Metadata ===")
    summary_marker = content.rindex("=== Global Summary ===")
    metadata_and_logs_block = content[metadata_start:summary_marker]
    metadata_lines: list[str] = []
    for line in metadata_and_logs_block.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") or stripped.startswith("==="):
            break
        if not line.strip():
            if metadata_lines:
                break
            continue
        if ":" not in line:
            continue
        metadata_lines.append(line)
    metadata_block = "\n".join(metadata_lines).strip()

    metadata: dict[str, object] = {}
    for line in metadata_block.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = _parse_scalar(value)

    summary_start = summary_marker + len("=== Global Summary ===")
    elapsed_marker = content.rindex("===仿真时间===")

    summary_block = content[summary_start:elapsed_marker].strip()
    elapsed_line = content[elapsed_marker:].splitlines()[0].strip()

    summary = ast.literal_eval(summary_block)
    log_line_summary = _parse_global_summary_log_line(content)
    for key, value in log_line_summary.items():
        if key not in summary or summary[key] is None:
            summary[key] = value
    elapsed_text = elapsed_line.replace("===仿真时间===", "").strip()
    if elapsed_text.endswith("s"):
        elapsed_text = elapsed_text[:-1].strip()
    elapsed_seconds = float(elapsed_text)

    return {
        **metadata,
        "elapsed_seconds": elapsed_seconds,
        "log_file_name": os.path.basename(log_file_path),
        "log_file_path": log_file_path,
        **summary,
    }


def collect_batch_log_rows(log_dir: str | None = None) -> list[dict[str, object]]:
    target_dir = log_dir or ensure_case_log_dir()
    if not os.path.isdir(target_dir):
        return []

    rows: list[dict[str, object]] = []
    for entry in sorted(os.listdir(target_dir)):
        if not entry.endswith(".log"):
            continue
        log_file_path = os.path.join(target_dir, entry)
        row = parse_batch_log(log_file_path)
        if row is not None:
            rows.append(row)
    return rows


def export_batch_log_summary_csv(
    rows: list[dict[str, object]],
    *,
    output_file_path: str,
) -> str:
    parameter_columns = [
        "label",
        "scenario_name",
        "scenario",
        "user_count",
        "arrival_rate_rps",
        "long_context_ratio",
        "long_context_prompt_tokens",
        "seed",
        "elapsed_seconds",
        "simulation_total_time_ms",
        "active_window_ms",
        "log_file_name",
        "log_file_path",
    ]
    throughput_columns = [
        "request_throughput_rps",
        "output_token_throughput_tps",
        "decode_token_throughput_tps",
        "prefill_token_throughput_tps",
    ]
    tpot_columns = [
        "request_tpot_avg_ms",
        "system_tpot_avg_ms",
        "requests_with_tpot_le_50ms",
        "requests_with_tpot_le_50ms_ratio",
    ]
    ttft_columns = [
        "prefill_first_token_latency_avg_ms",
        "prefill_first_token_latency_max_ms",
        "requests_with_prefill_first_token_latency_le_2s",
        "requests_with_prefill_first_token_latency_le_2s_ratio",
    ]
    utilization_columns = [
        "short_prefill_utilization",
        "long_prefill_utilization",
        "decode_utilization",
    ]

    prioritized_columns = (
        parameter_columns
        + throughput_columns
        + tpot_columns
        + ttft_columns
        + utilization_columns
    )
    percentile_suffix_tokens = ("_p50_", "_p95_", "_p99_")
    excluded_columns = {
        "tpot_ms",
        "ttft_ms",
        "req_queued_count",
        "req_queued_ratio",
    }
    all_columns: list[str] = []
    seen: set[str] = set()

    def should_export_column(column: str) -> bool:
        if column in excluded_columns:
            return False
        if column in prioritized_columns:
            return True
        return not any(token in column for token in percentile_suffix_tokens)

    for column in prioritized_columns:
        if column not in seen:
            all_columns.append(column)
            seen.add(column)
    for row in rows:
        for column in row.keys():
            if not should_export_column(column):
                continue
            if column not in seen:
                all_columns.append(column)
                seen.add(column)

    with open(output_file_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=all_columns,
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return output_file_path
