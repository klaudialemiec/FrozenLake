from stable_baselines3 import PPO
from stable_baselines3.common.type_aliases import GymEnv


class PPO_Agent(PPO):
    """
    The extension of Proximal Policy Optimization implementation 
    (https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html).
    It includes model params getter (used for storing data in mlflow).
    """

    def __init__(self, policy: str, env: GymEnv, **kwargs):
        super(PPO_Agent, self).__init__(policy, env, **kwargs)

    def get_params(self) -> dict:
        return {
            "learning_rate": self.learning_rate,
            "n_steps": self.n_steps,
            "n_epochs": self.n_epochs,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_range_vf": self.clip_range_vf,
            "ent_coef": self.ent_coef,
            "vf_coef": self.vf_coef,
            "max_grad_norm": self.max_grad_norm,
            "use_sde": self.use_sde,
            "sde_sample_freq": self.sde_sample_freq,
            "target_kl": self.target_kl,
        }
