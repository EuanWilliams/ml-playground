"""App for testing automated controllers for Pong"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import numpy as np

from pong.app import App
from pong.controllers import Controller
from pong.game.game import Game, Player, Level


@dataclass
class LifeLostLog:
    life: int
    level: Level
    paddle_bounce: int
    ball_count: int
    ball_x_positions: List[float]
    last_life: bool


class TestApp(App):
    def __init__(self, random: np.random.Generator, controllers: Dict[Player, Controller]) -> None:

        # TODO: singleplayer
        game = Game(random, single_player=True, logger=None)
        super().__init__(game, controllers, None)
        self.life_lost_logs: List[LifeLostLog] = []

    def _get_controller_extra_data(self) -> Optional[Dict[str, Any]]:
        """Gets any extra data required for any of the controllers"""
        return None

    def _render(self) -> None:
        """Does nothing."""
        return None

    def _game_over(self) -> None:
        pass

    def _tick(self, actions: Dict[Player, Any]) -> None:
        """Ticks the game forward by one tick, allows users to override functionality."""

        # TODO: single player
        self.left_lives_before = self.game.left_lives

        # tick
        super()._tick(actions)  # Call the parent class's _tick method to advance the game state

        self.left_lives_after = self.game.left_lives

        if self.left_lives_after != self.left_lives_before:
            ball_x_positions = [ball.position_x for ball in self.game.balls]
            self.life_lost_logs.append(
                LifeLostLog(
                    life=self.left_lives_before,
                    level=self.game.level,
                    paddle_bounce=self.game.paddle_bounces,
                    ball_count=len(self.game.balls),
                    ball_x_positions=ball_x_positions,
                    last_life=self.left_lives_after == 0,
                )
            )
