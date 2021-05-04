from stable_baselines3 import A2C
from stable_baselines3.common.type_aliases import GymEnv


class A2C_Agent(A2C):
    """
    The extension of Advantage Actor Critic implementation 
    (https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html).
    It includes model params getter (used for storing data in mlflow).
    """

    def __init__(self, policy: str, env: GymEnv, **kwargs):
        super(A2C_Agent, self).__init__(policy, env, **kwargs)

    def get_params(self) -> dict:
        return {
            "learning_rate": self.learning_rate,
            "n_steps": self.n_steps,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "ent_coef": self.ent_coef,
            "vf_coef": self.vf_coef,
            "max_grad_norm": self.max_grad_norm,
            "use_sde": self.use_sde,
            "sde_sample_freq": self.sde_sample_freq,
            "normalize_advantage": self.normalize_advantage,
        }
