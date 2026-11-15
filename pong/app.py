"""Base class for an app which implements a Gam"""
from abc import abstractmethod, ABC
from typing import Dict, Any, Optional

from pong.controllers import Controller
from pong.game.game import Game, Player, GameLogger, PaddleMove


class App(ABC):
    """Base class for an app, which is runs the game"""

    def __init__(self, game: Game, controllers: Dict[Player, Controller], logger: Optional[GameLogger]) -> None:
        self.game = game
        self.logger = logger
        self.single_player = self.game.single_player

        self.controllers = controllers

        if self.single_player and len(self.controllers) > 1:
            raise ValueError("Multiple controllers provided in single player mode")
        if not self.single_player and len(self.controllers) != 2:
            raise ValueError("Single controller provided in multiplayer mode")

    def run_game_to_completion(self) -> None:
        """Runs game to completion"""

        while self.game.winner is None:
            actions = self._retrieve_controller_actions()

            self.game.tick(actions)
            self._render()

        self._game_over()

    def _retrieve_controller_actions(self) -> Dict[Player, PaddleMove]:
        """Retrieves the controller's actions"""

        controller_extra_data = self._get_controller_extra_data()

        actions = {}
        for player, controller in self.controllers.items():
            controller.read_observation(self.game, extra_data=controller_extra_data)
            action = controller.get_move()
            actions[player] = action

        return actions

    @abstractmethod
    def _get_controller_extra_data(self) -> Optional[Dict[str, Any]]:
        """Gets any extra data required for any of the controllers, does nothing by default, optionally implemented by the child class"""

    @abstractmethod
    def _render(self) -> None:
        """Does nothing by default, can be implemented optionally by the child classes to render a frame to video or TUI"""

    @abstractmethod
    def _game_over(self) -> None:
        """Handles a game over event, does nothing by default can be implemented optionally by the child classes to render a frame to video or TUI"""
