"""A ball for pong"""

import math
import random
from typing import Tuple

import numpy as np
from numpy import typing as npt

from pong.constants import GRID_SIZE_Y, GRID_SIZE_X, MAX_STARTING_ANGLE, STARTING_VELOCITY_SCALAR, MAX_BALL_VELOCITY


class Ball:
    """The ball"""

    def __init__(self) -> None:
        self.position_x = 0.0
        self.position_y = 0.0

        self.velocity_y = 0.0
        self.velocity_x = 0.0
        self.reset_to_start()

    @property
    def position(self) -> Tuple[float, float]:
        """Returns the balls position"""
        return self.position_y, self.position_x

    @property
    def rounded_position(self) -> Tuple[int, int]:
        """Returns the balls position, floored to the current position."""
        return math.floor(self.position_y), math.floor(self.position_x)

    @property
    def next_position(self) -> Tuple[int, int]:
        """Returns the ball's next position, without actually moving the ball, floored to an integer."""
        return math.floor(self.position_y + self.velocity_y), math.floor(self.position_x + self.velocity_x)

    def tick(self) -> None:
        """Increments the balls position by one tick"""
        self.position_y = self.position_y + self.velocity_y
        self.position_x = self.position_x + self.velocity_x

    def bounce(self, unit_normal_vector: npt.NDArray[np.uint8]) -> None:
        """Bounces the ball, calculates the resulting velocity of the ball after the bounce.
        Calculates resulting velocity using the equation:

                vout = vin - 2(vin . N)N

        Where N is the unit normal vector of the surface.
        """

        # Calculate new velocity
        input_velocity = np.array([self.velocity_y, self.velocity_x], dtype=np.float32)
        self.velocity_y, self.velocity_x = input_velocity - (2 * np.dot(input_velocity, unit_normal_vector) * unit_normal_vector)

        # Send the ball off in the new velocity
        self.position_y = self.position_y + self.velocity_y
        self.position_x = self.position_x + self.velocity_x

    def accelerate(self, delta_y: float, delta_x: float) -> None:
        """Accelerates the ball a provided amount"""

        velocity_y = self.velocity_y + math.copysign(delta_y, self.velocity_y)
        velocity_x = self.velocity_x + math.copysign(delta_x, self.velocity_x)

        self.velocity_y = min(max(-MAX_BALL_VELOCITY, velocity_y), MAX_BALL_VELOCITY)
        self.velocity_x = min(max(-MAX_BALL_VELOCITY, velocity_x), MAX_BALL_VELOCITY)

    def set_position(self, position: Tuple[int, int]) -> None:
        """Sets the balls position"""
        self.position_y, self.position_x = position

    def reset_to_start(self) -> None:
        """Sets the ball to the middle of the frame and randomises starting velocity"""

        angle = random.uniform(-MAX_STARTING_ANGLE, MAX_STARTING_ANGLE)
        direction = random.choice((-1, 1))
        # y velocity in positions per tick - positive means moving top to bottom
        self.velocity_y = STARTING_VELOCITY_SCALAR * math.sin(angle)
        # x velocity in positions per tick - positive means moving left to right
        self.velocity_x = direction * STARTING_VELOCITY_SCALAR * math.cos(angle)

        self.position_y = GRID_SIZE_Y // 2
        self.position_x = GRID_SIZE_X // 2
