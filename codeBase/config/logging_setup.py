
import logging
from colorlog import ColoredFormatter

def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = ColoredFormatter(
            fmt="%(log_color)s[%(levelname)s] %(message)s",
            log_colors={
                "DEBUG":    "cyan",
                "INFO":     "green",
                "WARNING":  "yellow",
                "ERROR":    "red",
                "CRITICAL": "bold_red",
            }
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

# config_loader.py
import os
import yaml

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

