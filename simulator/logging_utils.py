from __future__ import annotations

import logging

from .config import LoggingConfig


class SimulationFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "sim_time_ms"):
            record.sim_time_ms = "-"
        return super().format(record)


def build_logger(config: LoggingConfig) -> logging.Logger:
    logger = logging.getLogger(config.logger_name)
    logger.setLevel(config.level)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    formatter = SimulationFormatter(
        fmt="[%(asctime)s][%(sim_time_ms)sms][%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if config.log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(config.level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if config.log_to_file and config.log_file_path:
        file_handler = logging.FileHandler(config.log_file_path, encoding="utf-8")
        file_handler.setLevel(config.level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if not logger.handlers:
        logger.addHandler(logging.NullHandler())

    return logger
