import json
from pathlib import Path
import os

vision_model = "ViT-B-16"
vision_model_config_file = \
    Path(__file__).parent / f"{vision_model.replace('/', '-')}.json"
print('Loading vision model config from', vision_model_config_file)
assert os.path.exists(vision_model_config_file)
with open(vision_model_config_file, 'r') as fv:
    model_info = json.load(fv).items()
    print('Model info:', model_info)
    if isinstance(model_info['vision_layers'], str):
        model_info['vision_layers'] = eval(model_info['vision_layers'])


