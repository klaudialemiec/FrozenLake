from stable_baselines3 import DQN
from stable_baselines3.common.type_aliases import GymEnv


class DQN_Agent(DQN):
    """
    The extension of Deep Q-Network implementation 
    (https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html).
    It includes model params getter (used for storing data in mlflow).
    """

    def __init__(self, policy: str, env: GymEnv, **kwargs):
        super(DQN_Agent, self).__init__(policy, env, **kwargs)

    def get_params(self) -> dict:
        return {
            "learning_rate": self.learning_rate,
            "buffer_size": self.buffer_size,
            "learning_starts": self.learning_starts,
            "batch_size": self.batch_size,
            "tau": self.tau,
            "gamma": self.gamma,
            "train_freq": self.train_freq,
            "gradient_steps": self.gradient_steps,
            "optimize_memory_usage": self.optimize_memory_usage,
            "target_update_interval": self.target_update_interval,
            "exploration_fraction": self.exploration_fraction,
            "exploration_initial_eps": self.exploration_initial_eps,
            "exploration_final_eps": self.exploration_final_eps,
            "max_grad_norm": self.max_grad_norm,
        }
