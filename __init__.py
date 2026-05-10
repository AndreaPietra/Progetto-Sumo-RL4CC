from .SumoEnv import SumoEnv
from ray.tune.registry import register_env

register_env("SumoEnv", lambda config: SumoEnv(config))
