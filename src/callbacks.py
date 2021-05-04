from pathlib import Path
from statistics import mean, median
from typing import Optional

import mlflow
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.utils import get_latest_run_id


class MlflowCallback(BaseCallback):
    """
    A custom callback saving infos to mlflow. Main usage to monitor training phase.
    """

    def __init__(self, model_path: Path):
        """
        :param model_path: Location where trained model will be saved
        :param tensorboard_path: tensorboard logs location
        """
        super(MlflowCallback, self).__init__()
        self.model_path = model_path

    def _on_training_start(self) -> None:
        """
        This method is called before at the very beginning of lerning.
        """
        mlflow.log_params(self.model.get_params())
        self.episodes_counter = 0

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        Saves episodic reward to mlflow.
        :return: wheater traning should be aborted early.
        """
        done = (
            self.locals["dones"][0]
            if "dones" in self.locals
            else self.locals["done"][0]
        )

        if done:
            reward = (
                self.locals["rewards"][0]
                if "rewards" in self.locals
                else self.locals["reward"][0]
            )
            mlflow.log_metric(
                "train_reward", reward, step=self.episodes_counter
            )
            self.episodes_counter += 1

        return True

    def _on_training_end(self) -> None:
        """
        Saves learned model, tensorboard logs (if exist) and learning timesteps.
        """
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(self.model_path)
        model_full_path = str(self.model_path) + ".zip"
        mlflow.log_artifact(model_full_path, "model")

        tensorboard_path = self._get_tensorboard_path()
        mlflow.log_artifact(tensorboard_path, "tensorboard")
        mlflow.log_param("learning steps", self.locals["total_timesteps"])

    def _get_tensorboard_path(self) -> Path:
        """
        :return: path to latest tensorboard directory.
        """
        tensorboard_parent_path = Path(self.model.tensorboard_log)
        last_tb_run_id = get_latest_run_id(
            tensorboard_parent_path, self.locals["tb_log_name"]
        )
        return (
            tensorboard_parent_path
            / f'{self.locals["tb_log_name"]}_{last_tb_run_id}'
        )


class MlflowEvalCallback(EvalCallback):
    """
    The extension of Evaluation Callback (https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback).
    It saves evaluation data to mlflow.
    """

    def __init__(
        self,
        eval_env: GymEnv,
        callback_on_new_best: Optional[BaseCallback] = None,
        n_eval_episodes: int = 10,
        **kwargs,
    ):
        super(MlflowEvalCallback, self).__init__(
            eval_env, callback_on_new_best, n_eval_episodes, **kwargs
        )

    def _on_training_end(self) -> None:
        mlflow.log_metric("eval_reward_mean", self.best_mean_reward)
