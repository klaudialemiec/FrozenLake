from pathlib import Path
from typing import List, Tuple

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import (
    CallbackList,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.type_aliases import GymEnv

from src.callbacks import MlflowCallback, MlflowEvalCallback


def train(
    model: BaseAlgorithm, timesteps: int, eval_env: GymEnv, model_path: Path
) -> None:
    """
    Train agent moves in his environment. Learning will finish when agent performs given number of timesteps or when mean reward of 10 gameplays reachs value 1.
    :param model: RL agent
    :param timesteps: total number of steps to take (through all episodes)
    :param eval_env: evaluation environment
    :param model_path: location where model will be saved
    :param tb_log_name: the name of the run for tensorboard log
    """
    mlflow_callback = MlflowCallback(model_path)
    reward_threshold_callback = StopTrainingOnRewardThreshold(
        reward_threshold=1
    )
    eval_callback = MlflowEvalCallback(
        eval_env=eval_env, callback_on_new_best=reward_threshold_callback
    )
    callbacks = CallbackList([mlflow_callback, eval_callback])

    model.learn(total_timesteps=timesteps, callback=callbacks)


def play(
    model: BaseAlgorithm,
    environment: GymEnv,
    episode_max_steps: int = 100,
    visualize: bool = True,
) -> Tuple[float, int]:
    """
    Plays and records one game in given environment.
    :param model: an agent taken actions
    :param environment: environment in which agent moves
    :param episode_max_steps: maximal number of agent moves in single episode
    :param gameplay_path: location in which gameplay will be recorded (as gif)
    :return: episode reward and number of moves taken to finish episode
    """
    images = []
    is_done = False
    step = 1
    state = environment.reset()

    while not is_done and step <= episode_max_steps:
        if visualize:
            environment.render()
            print()
        action, _ = model.predict(state)
        state, reward, is_done, info = environment.step(action)
        step += 1

    return reward, step
