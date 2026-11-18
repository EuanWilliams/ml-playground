"""For training a Pong model using reinforcement learning."""

import argparse
import datetime
import os
import json
from typing import List

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import LinearSchedule, ConstantSchedule
from stable_baselines3.common.vec_env import DummyVecEnv

from pong.game.game_constants import RANDOM_SEED
from pong.training.training_environment import (
    PongTrainingEnvironment,
    NO_EVENT_REWARD,
    LEVEL_EASY_LIFE_LOST_REWARD,
    LEVEL_MEDIUM_LIFE_LOST_REWARD,
    LEVEL_HARD_LIFE_LOST_REWARD,
    LEVEL_PADDLE_BOUNCE_EASY_REWARD,
    LEVEL_PADDLE_BOUNCE_MEDIUM_REWARD,
    LEVEL_PADDLE_BOUNCE_HARD_REWARD,
)

MODEL_ROOT_SAVE_DIR = "temp_models"
MODEL_SAVE_PREFIX = "pong_model_it_"
MODEL_VERSION_FILE_NAME = "model_version.json"

TENSORBOARD_PATH = "./tensorboard_logs"

# --- Hyperparams ---
NUMBER_OF_ENVIRONMENTS = 8  # Number of parallel environments to run for training
TRAINING_STEPS = 20_000_000
# From online: - A useful rule of thumb is that the agent effectively “sees” about 1 / (1 − γ) ticks ahead  where γ is gamma, for example, this means with a gamma of 0.995,
# the agent sees about 200 ticks ahead.
GAMMA = 0.990

EVALUATION_FREQUENCY = 100_000 // NUMBER_OF_ENVIRONMENTS
EVALUATION_EPISODES = 100

# n_steps is the PPO parameter that controls how many steps to run for each environment per update. The number of steps per update is n_steps * n_envs.
N_STEPS_RAW = 2048
N_STEPS = N_STEPS_RAW // NUMBER_OF_ENVIRONMENTS  # Adjust n_steps based on the number of environments to keep the total steps per update constant

ENABLE_EARLY_STOPPING = False  # Whether to enable early stopping based on evaluation performance
if ENABLE_EARLY_STOPPING:
    EARLY_STOPPING_PATIENCE = 10  # Number of evaluations with no improvement to wait before stopping training
    EARLY_STOPPING_MIN_EVALUATIONS = 20  # Minimum number of evaluations before early stopping can be triggered

LEARNING_RATE_APPROACH = "linear"  # Options: "constant", "linear"

if LEARNING_RATE_APPROACH == "linear":
    STARTING_LEARNING_RATE = 0.0003  # Starting learning rate for the optimizer
    ENDING_LEARNING_RATE = 0.00001  # Ending learning rate for the optimizer if using linear decay
    LEARNING_RATE_END_FRACTION = 0.90  # Fraction of training steps after which the learning rate reaches the ending value if using linear decay
elif LEARNING_RATE_APPROACH == "constant":
    CONSTANT_LEARNING_RATE = 0.0003  # Constant learning rate for the optimizer


def save_model_version(model: PPO, model_save_dir: str, iteration: int) -> None:
    """Builds the model version dictionary"""

    if LEARNING_RATE_APPROACH == "constant":
        learning_rate = {
            "approach": "constant",
            "start": CONSTANT_LEARNING_RATE,
            "end": CONSTANT_LEARNING_RATE,
        }
    elif LEARNING_RATE_APPROACH == "linear":
        learning_rate = {
            "approach": "linear",
            "start": STARTING_LEARNING_RATE,
            "end": ENDING_LEARNING_RATE,
            "end_fraction": LEARNING_RATE_END_FRACTION,
        }
    else:
        raise ValueError(f"Invalid learning rate approach: {LEARNING_RATE_APPROACH}")

    if ENABLE_EARLY_STOPPING:
        early_stopping = {
            "enabled": True,
            "patience": EARLY_STOPPING_PATIENCE,
            "min_evaluations": EARLY_STOPPING_MIN_EVALUATIONS,
            "used": model.num_timesteps <= TRAINING_STEPS,
        }
    else:
        early_stopping = {
            "enabled": False,
        }

    model_version = {
        "iteration": iteration,
        "trained_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "trained_steps": model.num_timesteps,
        "target_steps": TRAINING_STEPS,
        "gamma": GAMMA,
        "learning_rate": learning_rate,
        "n_environments": NUMBER_OF_ENVIRONMENTS,
        "n_steps": N_STEPS,
        "eval": {
            "frequency": EVALUATION_FREQUENCY,
            "episodes": EVALUATION_EPISODES,
        },
        "reward": {
            "level_easy_life_lost": LEVEL_EASY_LIFE_LOST_REWARD,
            "level_medium_life_lost": LEVEL_MEDIUM_LIFE_LOST_REWARD,
            "level_hard_life_lost": LEVEL_HARD_LIFE_LOST_REWARD,
            "level_easy_paddle_bounce": LEVEL_PADDLE_BOUNCE_EASY_REWARD,
            "level_medium_paddle_bounce": LEVEL_PADDLE_BOUNCE_MEDIUM_REWARD,
            "level_hard_paddle_bounce": LEVEL_PADDLE_BOUNCE_HARD_REWARD,
            "no_event": NO_EVENT_REWARD,
        },
        "early_stopping": early_stopping,
    }

    model_version_file_path = os.path.join(model_save_dir, MODEL_VERSION_FILE_NAME)
    with open(model_version_file_path, "w") as f:
        json.dump(model_version, f, indent=4)


def init_callbacks(model_save_dir: str) -> List[BaseCallback]:
    """Initializes the callbacks for training"""

    # TODO: test normalising the rewards.

    if ENABLE_EARLY_STOPPING:
        stop_train_callback = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=EARLY_STOPPING_PATIENCE, min_evals=EARLY_STOPPING_MIN_EVALUATIONS, verbose=1
        )
    else:
        stop_train_callback = None

    eval_env = Monitor(PongTrainingEnvironment(random_start=False))
    eval_env.reset(seed=RANDOM_SEED + 1)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_save_dir, "best"),
        log_path=os.path.join(model_save_dir, "eval_logs"),
        eval_freq=EVALUATION_FREQUENCY,
        n_eval_episodes=EVALUATION_EPISODES,
        deterministic=True,
        callback_after_eval=stop_train_callback,
    )

    return [eval_callback]


def prepare_learning_rate_schedule() -> Schedule:
    """Prepares the learning rate schedule based on the selected approach"""

    if LEARNING_RATE_APPROACH == "constant":
        return ConstantSchedule(CONSTANT_LEARNING_RATE)
    elif LEARNING_RATE_APPROACH == "linear":
        return LinearSchedule(start=STARTING_LEARNING_RATE, end=ENDING_LEARNING_RATE, end_fraction=LEARNING_RATE_END_FRACTION)
    else:
        raise ValueError(f"Invalid learning rate approach: {LEARNING_RATE_APPROACH}")


def initialise_fresh_model() -> PPO:
    """Initializes a fresh PPO model"""

    print("Initializing a fresh model.")
    env = make_vec_env(PongTrainingEnvironment, n_envs=NUMBER_OF_ENVIRONMENTS, seed=RANDOM_SEED, vec_env_cls=DummyVecEnv, env_kwargs={"random_start": True})
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=N_STEPS,
        verbose=1,
        gamma=GAMMA,
        tensorboard_log=TENSORBOARD_PATH,
        seed=RANDOM_SEED,
        learning_rate=prepare_learning_rate_schedule(),
    )
    return model


def initialise_model_from_weights(weights_path: str) -> PPO:
    """Initializes a PPO model from existing weights"""

    print(f"Loading model from weights: {weights_path}")
    env = make_vec_env(PongTrainingEnvironment, n_envs=NUMBER_OF_ENVIRONMENTS, seed=RANDOM_SEED, vec_env_cls=DummyVecEnv)
    model = PPO.load(
        weights_path, env=env, learning_rate=prepare_learning_rate_schedule(), n_steps=N_STEPS, gamma=GAMMA, tensorboard_log=TENSORBOARD_PATH, seed=RANDOM_SEED
    )
    return model


def main(opt: argparse.Namespace) -> None:
    """Main function"""

    model_output_path = MODEL_SAVE_PREFIX + str(opt.iteration)
    model_save_dir = os.path.join(MODEL_ROOT_SAVE_DIR, model_output_path)

    if os.path.exists(model_save_dir):
        raise FileExistsError(f"Model save directory {model_save_dir} already exists. Please choose a different iteration number.")

    os.makedirs(model_save_dir, exist_ok=False)
    model_save_path = os.path.join(model_save_dir, model_output_path)

    if opt.from_weights:
        model = initialise_model_from_weights(opt.from_weights)
    else:
        model = initialise_fresh_model()

    model.learn(total_timesteps=TRAINING_STEPS, tb_log_name=f"iteration_{opt.iteration}", progress_bar=True, callback=init_callbacks(model_save_dir))
    model.save(model_save_path)

    save_model_version(model, model_save_dir, iteration=opt.iteration)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training script for Pong")
    parser.add_argument("--iteration", type=int, required=True, help="Iteration number")
    parser.add_argument(
        "--from-weights", type=str, default=None, help="Path to the weights file to load for training. If not provided, a fresh model will be initialized."
    )
    args = parser.parse_args()

    main(args)
