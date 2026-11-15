"""Script for testing automated models at pong"""
import argparse
import statistics
from typing import Dict

import numpy as np
from tqdm import tqdm

from pong.controllers import ModelController, Controller, AutoController
from pong.game.game import Player
from pong.game.game_constants import RANDOM_SEED
from pong.training.test_app import TestApp

TEST_ITERATIONS = 100

RANDOM = np.random.default_rng(RANDOM_SEED)

def main(opt: argparse.Namespace) -> None:
    # controllers: Dict[Player, Controller] = {Player.LEFT: AutoController(None)}
    controllers = {Player.LEFT: ModelController(None)}

    print(f"Testing controller with {TEST_ITERATIONS} games...")
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
    args = parser.parse_args()

    main(args)
