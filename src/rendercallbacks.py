import os
import numpy as np
import imageio.v2 as imageio
from datetime import datetime
from ray.rllib.algorithms.callbacks import DefaultCallbacks


class VideoCallback(DefaultCallbacks):
  def __init__(self):
    super().__init__()
    self.save_dir = None
    self.global_training_episode_counter = 0
    self.global_evaluation_episode_counter = 0

  def on_episode_start(
      self, *, worker, base_env, episode, env_index, **kwargs
    ):
    # increment counter(s)
    if base_env.envs[0].is_evaluation:
      self.global_evaluation_episode_counter += 1
    else:
      self.global_training_episode_counter += 1
    # ensure the folder to save videos exists
    if self.save_dir is None:
      self.save_dir = os.path.join(
        base_env.envs[0].exp_logdir, "videos"
      )
      os.makedirs(self.save_dir, exist_ok=True)
    # only record some episodes
    self.max_frames = base_env.envs[0].render_max_frames
    self.record_this_episode = False
    if base_env.envs[0].is_evaluation:
      ene = base_env.envs[0].render_every_n_evaluation_episodes
      self.record_this_episode = (
        ene > 0 and self.global_evaluation_episode_counter % ene == 0
      )
    else:
      ene = base_env.envs[0].render_every_n_training_episodes
      self.record_this_episode = (
        ene > 0 and self.global_training_episode_counter % ene == 0
      )
    if self.record_this_episode:
      episode.user_data["frames"] = []

  def on_episode_step(
      self, *, worker, base_env, episode, env_index, **kwargs
    ):
    frames = episode.user_data.get("frames")
    if frames is None or len(frames) >= self.max_frames:
      return
    # get underlying gymnasium env; your SumoEnv is the first in the vector
    env = base_env.envs[0]
    # render_mode must have been set to "rgb_array" in env_config
    frame = env.render()
    if frame is not None:
      frames.append(frame)

  def on_episode_end(
      self, *, worker, base_env, policies, episode, env_index, **kwargs
    ):
    frames = episode.user_data.get("frames")
    if frames is None:
      return
    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')
    key = "eval" if base_env.envs[0].is_evaluation else "train"
    path = os.path.join(self.save_dir, f"{key}_episode_{now}.gif")
    # frames is a list of H×W×3 uint8 arrays
    imageio.mimsave(path, frames, fps = 10)
