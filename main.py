import src
from RL4CC.utilities.common import load_config_file
from datetime import datetime
import os
from RL4CC.experiments.train import TrainingExperiment



exp = TrainingExperiment(exp_config_file="exp_config.json")
exp.run()