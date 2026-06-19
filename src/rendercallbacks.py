# callbacks.py
import os
import numpy as np
import imageio.v2 as imageio
from ray.rllib.algorithms.callbacks import DefaultCallbacks

class VideoCallback(DefaultCallbacks):
    def __init__(self, save_dir="videos", every_n_episodes=1, max_frames=300):
        super().__init__()
        self.save_dir = save_dir
        self.every_n_episodes = every_n_episodes
        self.max_frames = max_frames
        os.makedirs(self.save_dir, exist_ok=True)

    def on_episode_start(self, *, worker, base_env, episode, env_index, **kwargs):
        # Only record some episodes
        if episode.episode_id % self.every_n_episodes == 0:
            episode.user_data["frames"] = []

    def on_episode_step(self, *, worker, base_env, episode, env_index, **kwargs):
        frames = episode.user_data.get("frames")
        if frames is None:
            return

        # Get underlying gymnasium env; your SumoEnv is the first in the vector
        env = base_env.envs[0]

        # render_mode must have been set to "rgb_array" in env_config
        frame = env.render()
        print(f"========= {env}")
        print(f"++++++++++ {frame}")
        if frame is not None:
            frames.append(frame)
            if len(frames) >= self.max_frames:
                # stop recording further frames for this episode
                episode.user_data["frames"] = None

    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        frames = episode.user_data.get("frames")
        if not frames:
            return

        ep_id = episode.episode_id
        path = os.path.join(self.save_dir, f"episode_{ep_id}.gif")

        # frames is a list of H×W×3 uint8 arrays
        imageio.mimsave(path, frames, fps=10)

