from ray.rllib.env.env_context import EnvContext
from sumo_rl.environment.env import SumoEnvironment
from RL4CC.environment.base_multiagent_environment import BaseMultiAgentEnvironment
from gymnasium.spaces import Dict
import os

class SumoEnv(BaseMultiAgentEnvironment, SumoEnvironment):
    def __init__(self, env_config: EnvContext):
        """
        Initializes SUMO simulation and builds observation/action spaces
        as Gymnasium Dict spaces keyed by traffic signal ID.
        """
        seed = self.load_configuration(env_config)
        self._action_space_in_preferred_format = True
        self._obs_space_in_preferred_format = True

        SumoEnvironment.__init__(
            self,
            net_file=env_config["net_file"],
            route_file=env_config["route_file"],
            use_gui=env_config.get("use_gui", False),
            begin_time=self.min_time,
            num_seconds=self.max_time,
            delta_time=self.time_step,
            reward_fn=env_config.get("reward_fn", "diff-waiting-time"),
            single_agent=False,
            sumo_seed=seed if seed is not None else "random",
            fixed_ts=env_config.get("fixed_ts", True),
            out_csv_name=self.out_csv_name,
            additional_sumo_cmd=env_config.get("additional_sumo_cmd"),
            render_mode=env_config.get("render_mode")
        )

        self._observation_space = Dict({
            ts: self.traffic_signals[ts].observation_space
            for ts in self.traffic_signals
        })
        self._action_space = Dict({
            ts: self.traffic_signals[ts].action_space
            for ts in self.traffic_signals
        })

    def load_configuration(self, env_config):
        seed = BaseMultiAgentEnvironment.load_configuration(self, env_config)
        self.out_csv_name = env_config.get("out_csv_name", "sumo_csv")
        if self.exp_logdir:
            self.out_csv_name = os.path.join(self.exp_logdir, self.out_csv_name, self.out_csv_name)
        self.is_evaluation = env_config.get("is_evaluation", False)
        # render configuration
        self.render_every_n_training_episodes = env_config.get(
          "render_every_n_training_episodes", -1
        )
        self.render_every_n_evaluation_episodes = env_config.get(
          "render_every_n_evaluation_episodes", -1
        )
        self.render_max_frames = env_config.get(
          "render_max_frames", (self.max_time-self.min_time) // self.time_step
        )
        return seed

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def _structure_info(self, infos: dict):
        """
        Converts SUMO's flat info dict into RLlib's nested format {agent_id: {metric: value}}.
        Keys not matching any agent are grouped under "__common__".
        """
        structured_info = {}
        for k, v in infos.items():
            added = False
            for agent in self.agents:
                if agent in k:
                    if agent not in structured_info:
                        structured_info[agent] = {}
                    structured_info[agent][k.replace(agent, "")] = v
                    added = True
            if not added:
                if "__common__" not in structured_info:
                    structured_info["__common__"] = {}
                structured_info["__common__"][k] = v
        return structured_info

    def reset(self, seed: int = None, **kwargs):
        obs = SumoEnvironment.reset(self, seed, **kwargs)
        infos = SumoEnvironment._compute_info(self)
        return obs, self._structure_info(infos)

    def step(self, action_dict):
        """
        Runs one simulation step. Returns a 5-tuple (obs, rewards, terminated,
        truncated, infos) as required by Gymnasium — dones is used for both
        terminated and truncated since SUMO does not distinguish them.
        """
        obs, rewards, dones, info = SumoEnvironment.step(self, action_dict)
        print(info)
        return obs, rewards, dones, dones, self._structure_info(info)

    def render(self):
        return SumoEnvironment.render(self)
