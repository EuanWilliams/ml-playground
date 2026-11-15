"""App for testing automated controllers for Pong"""

from typing import Optional, Dict, Any

import numpy as np

from pong.app import App
from pong.controllers import Controller
from pong.game.game import Game, Player


class TestApp(App):
    def __init__(self, random: np.random.Generator, controllers: Dict[Player, Controller]) -> None:

        # TODO: singleplayer
        game = Game(random, single_player=True, logger=None)
        super().__init__(game, controllers, None)

    def _get_controller_extra_data(self) -> Optional[Dict[str, Any]]:
        """Gets any extra data required for any of the controllers"""
        return None

    def _render(self) -> None:
        """Does nothing."""
        return None

    def _game_over(self) -> None:
        pass
