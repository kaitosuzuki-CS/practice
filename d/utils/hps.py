import json


class HPS:
    def __init__(self, hps):
        for key, value in hps.items():
            if isinstance(value, dict):
                setattr(self, key, HPS(value))
            else:
                setattr(self, key, value)


def load_hps(path):
    with open(path, "r") as f:
        data = json.load(f)

    model_config = data["model_config"]
    train_config = data["train_config"]
    inference_config = data["inference_config"]

    model_hps = HPS(model_config)
    train_hps = HPS(train_config)
    inference_hps = HPS(inference_config)

    return model_hps, train_hps, inference_hps
