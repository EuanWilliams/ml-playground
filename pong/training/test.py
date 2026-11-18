"""Script for testing automated models at pong"""

import argparse
import os
import statistics
from typing import Dict, Optional, List, Any

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from pong.controllers import ModelController, Controller, AutoController
from pong.game.game import Player
from pong.game.game_constants import RANDOM_SEED, LEVEL_EASY_MAX_BOUNCES, LEVEL_MEDIUM_MAX_BOUNCES, LEVEL_HARD_NEW_BALLS_PADDLE_BOUNCES
from pong.training.test_app import TestApp

TEST_ITERATIONS = 100

MODE_AUTO = "auto"
MODE_MODEL = "model"

FINAL_MODEL_PATH = "pong/models/"
TEMP_MODEL_PATH = "temp_models/"
LATEST_MODEL_PATH = "pong_model_it_{ITERATION}/pong_model_it_{ITERATION}"
BEST_MODEL_PATH = "pong_model_it_{ITERATION}/best/best_model"

RANDOM = np.random.default_rng(RANDOM_SEED)

def plot_background_levels(ax: plt.Axes, all_paddle_bounces: List[int]) -> None:
    """Plots the background levels for the different levels of the game."""

    # overlay a shaded background for each level:
    ax.axvspan(0, LEVEL_EASY_MAX_BOUNCES, color="green", alpha=0.1, label="Level Easy")
    ax.axvspan(LEVEL_EASY_MAX_BOUNCES, LEVEL_MEDIUM_MAX_BOUNCES, color="yellow", alpha=0.1, label="Level Medium")
    ax.axvspan(LEVEL_MEDIUM_MAX_BOUNCES, LEVEL_MEDIUM_MAX_BOUNCES + LEVEL_HARD_NEW_BALLS_PADDLE_BOUNCES, color="red", alpha=0.1, label="Level Hard (+ 1 ball)")

    # overlay as shaded background for each new ball added in level hard:
    colours = ["purple", "orange", "cyan", "magenta", "brown", "pink", "gray", "olive", "teal", "navy"]
    step = LEVEL_HARD_NEW_BALLS_PADDLE_BOUNCES
    for n, start in enumerate(range(LEVEL_MEDIUM_MAX_BOUNCES, max(all_paddle_bounces), step)):
        ax.axvspan(start, start + step, color=colours[n % len(colours)], alpha=0.1, label=f"New Ball {n + 2}")

def plot_all_lives_lost(life_lost_data: List[Any], plot_output_path: Optional[str]) -> None:
    """Plot the distribution of all life lost events over paddle bounces. This is informative as it shows where the controller most commonly loses lives."""

    # plot the standard distribution of life lost events over paddle bounces
    all_paddle_bounces = [log.paddle_bounce for logs in life_lost_data for log in logs]
    plt.hist(all_paddle_bounces, bins=range(0, max(all_paddle_bounces) + 1, 1), density=True, alpha=0.5, color="blue")

    plot_background_levels(plt.gca(), all_paddle_bounces)

    plt.title("Distribution of Life Lost Events Over Paddle Bounces")
    plt.xlabel("Paddle Bounces")
    plt.ylabel("Density")
    plt.legend()
    if plot_output_path is not None:
        output_path = os.path.join(plot_output_path, "life_lost_distribution.png")
        plt.savefig(output_path)
    plt.show()

def plot_last_lives_lost(life_lost_data: List[Any], plot_output_path: Optional[str]) -> None:
    """Plot the distribution of last life lost events over paddle bounces. This is informative as it shows where the controller most commonly loses all its
    lives"""

    last_life_paddle_bounces = [log.paddle_bounce for logs in life_lost_data for log in logs if log.last_life]
    plt.hist(last_life_paddle_bounces, bins=range(0, max(last_life_paddle_bounces) + 1, 1), density=True, alpha=0.5, color="blue")
    plot_background_levels(plt.gca(), last_life_paddle_bounces)

    plt.title("Distribution of Last Life Lost Events Over Paddle Bounces")
    plt.xlabel("Paddle Bounces")
    plt.ylabel("Density")
    plt.legend()
    if plot_output_path is not None:
        output_path = os.path.join(plot_output_path, "last_life_lost_distribution.png")
        plt.savefig(output_path)
    plt.show()

def main(mode: str, iteration: Optional[int], plot_output_path: str) -> None:
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
    life_lost_data = []
    for _ in tqdm(range(TEST_ITERATIONS)):
        app = TestApp(RANDOM, controllers)
        app.run_game_to_completion()
        life_lost_data.append(app.life_lost_logs)
        scores.append(app.game.score)

    print("Test complete.")
    print(f"Results over {TEST_ITERATIONS} games.")
    print(f"Average score: {statistics.mean(scores)}")
    print(f"Median score: {statistics.median(scores)}")
    print(f"Standard deviation: {statistics.stdev(scores)}")
    print(f"Min score achieved: {min(scores)}")
    print(f"Max score achieved: {max(scores)}")

    print("Life lost data:")
    print("Average level reached before life lost:", statistics.mean([log.level.value for logs in life_lost_data for log in logs]))
    print("Average paddle bounces before life lost:", statistics.mean([log.paddle_bounce for logs in life_lost_data for log in logs]))
    print("Average ball count before life lost:", statistics.mean([log.ball_count for logs in life_lost_data for log in logs]))

    # --- plot - standard distribution of when lives are lost broken down by level

    plot_all_lives_lost(life_lost_data, plot_output_path)

    #  --- plot - standard distribution of last life lost events over paddle bounces broken down by level
    plot_last_lives_lost(life_lost_data, plot_output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Testing script for testing automated models at pong.")
    parser.add_argument("--mode", type=str, default=MODE_MODEL, choices=[MODE_AUTO, MODE_MODEL], help="Which mode to play in.")
    parser.add_argument("--iteration", type=int, default=None, help="The iteration of the model to test.")
    parser.add_argument("--best", action="store_true", help="Whether to use the best model or the latest model.")
    parser.add_argument(
        "--interim", action="store_true", help="Whether the model is an interim model stored in temp_models/ or a final model stored in pong/models/"
    )
    parser.add_argument("--plot-output-path", type=str, default=None, help="Path to save the plots to, leave blank to not save the plots.")
    args = parser.parse_args()

    if args.mode == MODE_MODEL and args.iteration is None:
        raise ValueError("Iteration required when testing in MODEL mode.")
    elif args.mode == MODE_AUTO and (args.iteration is not None or args.best or args.interim):
        raise ValueError("Iteration not required when testing in AUTO mode.")

    main(args.mode, args.iteration, args.plot_output_path)
