"""Pong paddle"""

from typing import Tuple

from pong.game.game_constants import GRID_SIZE_Y


class Paddle:
    """A paddle"""

    def __init__(self, starting_y: int, x_position: int) -> None:
        self.position_y = starting_y
        self.position_x = x_position

    @property
    def position(self) -> Tuple[int, int]:
        """Returns the paddle position in format y, x"""
        return self.position_y, self.position_x

    def move_paddle(self, increment: int) -> None:
        """Moves the ball a single increment"""
        self.position_y = max(min(self.position_y + increment, GRID_SIZE_Y - 2), 1)

    def hits_paddle(self, ball_position_y: int, ball_position_x: int) -> bool:
        """Checks if the ball collides with the paddle"""
        return ball_position_y == self.position_y and ball_position_x == self.position_x
