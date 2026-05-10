from ray.rllib.env.env_context import EnvContext
from sumo_rl.environment.env import SumoEnvironment
from RL4CC.environment.base_multiagent_environment import BaseMultiAgentEnvironment
from gymnasium.spaces import Dict
import os


class SumoEnv(BaseMultiAgentEnvironment, SumoEnvironment):

    def __init__(self, env_config: EnvContext):

        # load configuration and define simulation length

        seed = self.load_configuration(env_config)
        self._action_space_in_preferred_format = True
        self._obs_space_in_preferred_format = True
        #num_seconds = (self.max_time - self.min_time) // self.time_step

        # initialize SUMO environment

        SumoEnvironment.__init__(
            self,
            net_file = env_config["net_file"],
            route_file = env_config["route_file"],
            use_gui = env_config.get("use_gui", False),
            begin_time = self.min_time,          
            num_seconds = self.max_time,        
            delta_time = self.time_step,
            reward_fn = env_config.get("reward_fn", "diff-waiting-time"),
            single_agent = False,
            sumo_seed = seed if seed is not None else "random",
            fixed_ts = env_config.get("fixed_ts", True),
            out_csv_name = self.out_csv_name

        )

        # -- extract observation and action spaces

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

        return seed
    
    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    

    def _structure_info(self, infos: dict):

        """
        Convert the info dictionary returned by SUMO environment in the 

        format required by RLLib
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

        """
        Reset SUMO environment and compute info

        """
        obs = SumoEnvironment.reset(self, seed, **kwargs)
        infos = SumoEnvironment._compute_info(self)

        return obs, self._structure_info(infos)


    def step(self, action_dict):
        """
        Compute next environment step

        """
        obs, rewards, dones, info = SumoEnvironment.step(self, action_dict)
        print(info)

        return obs, rewards, dones, dones, self._structure_info(info)