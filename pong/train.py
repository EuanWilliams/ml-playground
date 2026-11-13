import argparse
from stable_baselines3 import PPO

from pong.constants import GAMMA, TRAINING_STEPS
from pong.training_environment import PongTrainingEnvironment

MODEL_SAVE_PREFIX = "pong_model"

TENSORBOARD_PATH = "./tensorboard_logs"


def main(opt: argparse.Namespace) -> None:
    """Main function"""

    env = PongTrainingEnvironment()
    model = PPO("MlpPolicy", env, verbose=1, gamma=GAMMA, tensorboard_log=TENSORBOARD_PATH)
    model.learn(total_timesteps=TRAINING_STEPS)
    model_output_path = MODEL_SAVE_PREFIX + f"_it_{opt.iteration}"
    model.save(model_output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training script for Pong")
    parser.add_argument("--iteration", type=int, required=True, help="Iteration number")
    args = parser.parse_args()

    main(args)
