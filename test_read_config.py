from DL.src.models.utils import _as_dict, _get_first
from DL.src.models.loading import load_config

cfg = load_config('DL/experiments/example_fusion_config.yaml')

print(type(cfg))