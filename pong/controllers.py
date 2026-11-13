"""Controllers are used for controlling paddles in pong"""

from dataclasses import dataclass
from typing import List, Optional, Protocol

from stable_baselines3 import PPO

from pong.constants import PADDLE_HALF_HEIGHT, MODEL_PATH
from pong.game import PaddleMove, GameLogger
from pong.model_common import ModelObservationT


class Controller(Protocol):
    def get_move(self) -> PaddleMove: ...


@dataclass
class Observation:
    ball_position_y: float
    ball_position_x: float
    ball_velocity_y: float
    paddle_position_y: int


class KeyboardController(Controller):
    """A keyboard controller for the user"""

    def __init__(self, key_press_up: List[int], key_press_down: List[int]) -> None:
        self.last_key_press: Optional[int] = None
        self.key_press_up = key_press_up
        self.key_press_down = key_press_down

    def add_keypress(self, keypress: Optional[int]) -> None:
        """Stores the keypress."""

        self.last_key_press = keypress

    def get_move(self) -> PaddleMove:
        """Returns the next move."""

        if self.last_key_press in self.key_press_up:
            return PaddleMove.UP
        elif self.last_key_press in self.key_press_down:
            return PaddleMove.DOWN
        else:
            return PaddleMove.STAY


class AutoController(Controller):
    """A simple auto controller. Takes the ball's current position and velocity and moves the paddle towards its predicted location."""

    def __init__(self, logger: GameLogger) -> None:
        self.last_observations: Optional[List[Observation]] = None
        self.logger = logger

    def add_observation(self, observation: List[Observation]) -> None:
        """Adds observation of the balls current position."""

        self.last_observations = observation

    def get_move(self) -> PaddleMove:
        """Selects the next move based on the last observed observation"""

        assert self.last_observations is not None

        target_observation = sorted(self.last_observations, key=lambda observation: observation.ball_position_x)[0]

        predicted_next_position_y = target_observation.ball_position_y + target_observation.ball_velocity_y
        delta_to_ball_next_pos = predicted_next_position_y - (target_observation.paddle_position_y + PADDLE_HALF_HEIGHT)

        if delta_to_ball_next_pos > 0.2:
            self.logger.add_log_message(
                f"Ball position is ball_position is {target_observation.ball_position_y}, ball_velocity is {target_observation.ball_velocity_y}. Paddle position is {target_observation.paddle_position_y}, moving paddle DOWN"
            )
            return PaddleMove.DOWN
        elif delta_to_ball_next_pos < -0.2:
            self.logger.add_log_message(
                f"Ball position is ball_position is {target_observation.ball_position_y}, ball_velocity is {target_observation.ball_velocity_y}. Paddle position is {target_observation.paddle_position_y}, moving paddle UP"
            )
            return PaddleMove.UP

        return PaddleMove.STAY


class ModelController(Controller):
    """Controller controlled by Ping - our RL model!"""

    def __init__(self, logger: GameLogger) -> None:
        self.last_observation: Optional[ModelObservationT] = None
        self.logger = logger
        self.model = PPO.load(MODEL_PATH)  # TODO: put in a wrapper

    def add_observation(self, observation: ModelObservationT) -> None:
        """Adds observation of the balls current position."""

        self.last_observation = observation

    def get_move(self) -> PaddleMove:
        """Predicts the next action."""
        assert self.last_observation is not None
        action, _states = self.model.predict(self.last_observation)
        return PaddleMove(int(action) - 1)  # have to shift action -1 as the models action space is 0, 1, 2 and our paddle moves are -1, 0, 1
