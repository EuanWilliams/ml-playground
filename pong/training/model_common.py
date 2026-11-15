import math

import numpy as np
from numpy import typing as npt

from pong.game.ball import Ball
from pong.game.game_constants import GRID_SIZE_Y, GRID_SIZE_X, MAX_BALLS, MAX_BALL_VELOCITY
from pong.game.game import Game

ModelObservationT = npt.NDArray[np.float32]
ModelActionT = np.int8


def time_to_reach_paddle(paddle_position_x: int, ball: Ball) -> float:
    """Estimated ticks until the ball reaches the left paddle's x position. Uses time = distance / speed to estimate the number of ticks until
    hit."""

    right_wall_x = GRID_SIZE_X - 1
    velocity_x = ball.velocity_x

    if velocity_x < 0:
        if ball.position_x < paddle_position_x:
            return math.inf  # already behind the paddle, cannot be saved so least relevant
        return (ball.position_x - paddle_position_x) / abs(velocity_x)

    if velocity_x > 0:
        return ((right_wall_x - ball.position_x) + (right_wall_x - paddle_position_x)) / abs(velocity_x)

    return math.inf  # no horizontal motion, never arrives


def get_observation(game: Game) -> ModelObservationT:
    """Returns an observation"""

    balls_by_proximity = sorted(game.balls, key=lambda ball: time_to_reach_paddle(game.left_paddle.position_x, ball))

    # The observation shape is [{paddle observation}, {closest_ball_observation}, {next_closest}, ....]
    observation_list = []

    # Paddle observation is: [position_y, position_x]
    observation_list.extend([game.left_paddle.position_y / GRID_SIZE_Y, game.left_paddle.position_x / GRID_SIZE_X])

    # Each balls observation is in format: [present, ball_position_y, ball_position_x, ball_velocity_y, ball_velocity_x]
    for _ in range(MAX_BALLS):
        if len(balls_by_proximity):
            ball = balls_by_proximity.pop(0)  # take the first one off the list, this is closest to the paddle.
            # Normalise these values here. They should always be between -1 and 1.
            present = 1
            ball_position_y = ball.position_y / GRID_SIZE_Y
            ball_position_x = ball.position_x / GRID_SIZE_X
            ball_velocity_y = ball.velocity_y / MAX_BALL_VELOCITY
            ball_velocity_x = ball.velocity_x / MAX_BALL_VELOCITY
        else:
            present = 0
            ball_position_y = 0
            ball_position_x = 0
            ball_velocity_y = 0
            ball_velocity_x = 0

        observation_list.extend([present, ball_position_y, ball_position_x, ball_velocity_y, ball_velocity_x])

    return np.array(observation_list, dtype=np.float32)
