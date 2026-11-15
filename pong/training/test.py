"""Script for testing automated models at pong"""

import argparse
import os
import statistics
from typing import Dict, Optional

import numpy as np
from tqdm import tqdm

from pong.controllers import ModelController, Controller, AutoController
from pong.game.game import Player
from pong.game.game_constants import RANDOM_SEED
from pong.training.test_app import TestApp

TEST_ITERATIONS = 100

MODE_AUTO = "auto"
MODE_MODEL = "model"

FINAL_MODEL_PATH = "pong/models/"
TEMP_MODEL_PATH = "temp_models/"
LATEST_MODEL_PATH = "pong_model_it_{ITERATION}/pong_model_it_{ITERATION}"
BEST_MODEL_PATH = "pong_model_it_{ITERATION}/best/best_model"

RANDOM = np.random.default_rng(RANDOM_SEED)


def main(mode: str, iteration: Optional[int]) -> None:
    print(f"Testing controller with {TEST_ITERATIONS} games...")
    if mode == MODE_AUTO:
        controllers: Dict[Player, Controller] = {Player.LEFT: AutoController(None)}
    elif mode == MODE_MODEL:
        model_root = FINAL_MODEL_PATH if not args.interim else TEMP_MODEL_PATH
        model_name = BEST_MODEL_PATH if args.best else LATEST_MODEL_PATH
        model_name = model_name.format(ITERATION=iteration)
        model_path = os.path.join(model_root, model_name)
        controllers = {Player.LEFT: ModelController(model_path)}
        print("Using model in path:", model_path)
    else:
        raise ValueError(f"Invalid mode: {mode}")

    scores = []
    for _ in tqdm(range(TEST_ITERATIONS)):
        app = TestApp(RANDOM, controllers)
        app.run_game_to_completion()
        scores.append(app.game.score)

    print("Test complete.")
    print(f"Results over {TEST_ITERATIONS} games.")
    print(f"Average score: {statistics.mean(scores)}")
    print(f"Median score: {statistics.median(scores)}")
    print(f"Standard deviation: {statistics.stdev(scores)}")
    print(f"Min score achieved: {min(scores)}")
    print(f"Max score achieved: {max(scores)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Testing script for testing automated models at pong.")
    parser.add_argument("--mode", type=str, default=MODE_MODEL, choices=[MODE_AUTO, MODE_MODEL], help="Which mode to play in.")
    parser.add_argument("--iteration", type=int, default=None, help="The iteration of the model to test.")
    parser.add_argument("--best", action="store_true", help="Whether to use the best model or the latest model.")
    parser.add_argument(
        "--interim", action="store_true", help="Whether the model is an interim model stored in temp_models/ or a final model stored in pong/models/"
    )
    args = parser.parse_args()

    if args.mode == MODE_MODEL and args.iteration is None:
        raise ValueError("Iteration required when testing in MODEL mode.")
    elif args.mode == MODE_AUTO and (args.iteration is not None or args.best or args.interim):
        raise ValueError("Iteration not required when testing in AUTO mode.")

    main(args.mode, args.iteration)
