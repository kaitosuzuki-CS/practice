import os

import yaml


class HPS:
    def __init__(self, hps):
        for key, value in hps.items():
            if isinstance(value, dict):
                setattr(self, key, HPS(value))
            else:
                setattr(self, key, value)


def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return HPS(config)
