"""Controllers are used for controlling paddles in pong"""

from dataclasses import dataclass
from typing import List, Optional, Protocol, Dict, Any

from stable_baselines3 import PPO

from pong.game.game_constants import PADDLE_HALF_HEIGHT
from pong.game.game import PaddleMove, GameLogger, Game
from pong.training.model_common import ModelObservationT, get_observation

# Scripted controller
AUTO_MOVE_DELTA_THRESHOLD = 0.2  # if the ball is within this distance of the paddle, don't move the paddle.

# Model controller
MODEL_PATH = "pong/models/pong_model_it_0"

class Controller(Protocol):
    def read_observation(self, game: Game, extra_data: Optional[Dict[str, Any]]) -> None: ...
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

    def read_observation(self, game: Game, extra_data: Optional[Dict[str, Any]]) -> None:
        """Reads the game and any additional information to determine the next move."""

        keystroke = extra_data.get("keystroke") if extra_data else None

        valid_keystroke = None
        if keystroke in [*self.key_press_up, *self.key_press_down]:
            valid_keystroke = keystroke

        self.last_key_press = valid_keystroke

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

    def read_observation(self, game: Game, extra_data: Optional[Dict[str, Any]]) -> None:
        """Reads the game and any additional information to determine the next move."""

        # TODO: only supports single player for now.
        observations = [Observation(ball.position_y, ball.position_x, ball.velocity_y, game.left_paddle.position_y) for ball in game.balls]
        self.last_observations = observations

    def get_move(self) -> PaddleMove:
        """Selects the next move based on the last observed observation"""

        assert self.last_observations is not None

        target_observation = sorted(self.last_observations, key=lambda observation: observation.ball_position_x)[0]

        predicted_next_position_y = target_observation.ball_position_y + target_observation.ball_velocity_y
        delta_to_ball_next_pos = predicted_next_position_y - (target_observation.paddle_position_y + PADDLE_HALF_HEIGHT)

        if delta_to_ball_next_pos > AUTO_MOVE_DELTA_THRESHOLD:
            self.logger.add_log_message(
                f"Ball position is ball_position is {target_observation.ball_position_y}, ball_velocity is {target_observation.ball_velocity_y}. Paddle position is {target_observation.paddle_position_y}, moving paddle DOWN"
            )
            return PaddleMove.DOWN
        elif delta_to_ball_next_pos < -AUTO_MOVE_DELTA_THRESHOLD:
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

    def read_observation(self, game: Game, extra_data: Optional[Dict[str, Any]]) -> None:
        """Reads the game and any additional information to determine the next move."""

        observation = get_observation(game)
        self.last_observation = observation

    def get_move(self) -> PaddleMove:
        """Predicts the next action."""
        assert self.last_observation is not None
        action, _states = self.model.predict(self.last_observation)
        return PaddleMove(int(action) - 1)  # have to shift action -1 as the models action space is 0, 1, 2 and our paddle moves are -1, 0, 1
