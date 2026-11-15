import argparse
import datetime
from typing import Dict, Any

from stable_baselines3 import PPO

from pong.training.training_environment import PongTrainingEnvironment

MODEL_SAVE_PREFIX = "pong_model_it_"
MODEL_VERSION_FILE_NAME = "model_version.json"

TENSORBOARD_PATH = "./tensorboard_logs"

TRAINING_STEPS = 2_000_000
GAMMA = 0.995  # From online: - A useful rule of thumb is that the agent effectively “sees” about 1 / (1 − γ) ticks ahead  where γ is gamma


def build_model_version(iteration: int, learning_rate: float) -> Dict[str, Any]:
    """Builds the model version dictionary"""

    return {
        "iteration": iteration,
        "trained_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "trained_steps": TRAINING_STEPS,
        "gamma": GAMMA,
        "learning_rate": learning_rate,
    }


def main(opt: argparse.Namespace) -> None:
    """Main function"""

    env = PongTrainingEnvironment()
    model = PPO("MlpPolicy", env, verbose=1, gamma=GAMMA, tensorboard_log=TENSORBOARD_PATH)
    model.learn(total_timesteps=TRAINING_STEPS)
    model_output_path = MODEL_SAVE_PREFIX + str(opt.iteration)

    build_model_version(iteration=opt.iteration, learning_rate=model.learning_rate)
    model.save(model_output_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training script for Pong")
    parser.add_argument("--iteration", type=int, required=True, help="Iteration number")
    args = parser.parse_args()

    main(args)
