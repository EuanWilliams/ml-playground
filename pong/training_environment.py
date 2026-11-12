"""Used for training the RL model"""

import math
from typing import Any, Dict, Tuple, Optional, SupportsFloat, List

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import numpy as np
import numpy.typing as npt

from pong.ball import Ball
from pong.constants import GRID_SIZE_Y, GRID_SIZE_X, MAX_BALLS, MAX_BALL_VELOCITY
from pong.game import Game, Player, PaddleMove, TickEvent, Level

ObservationT = npt.NDArray[np.float32]
ActionT = np.int8


class TrainingGame(gym.Env):
    def __init__(self) -> None:
        super().__init__()

        self.grid_size_y = GRID_SIZE_Y
        self.grid_size_x = GRID_SIZE_X

        shape = 2 + (MAX_BALLS * 5)
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(shape,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3, start=0, dtype=np.int8)

        self.game = Game(self.np_random, single_player=True, logger=None)  # Start with a fresh game

    def _time_to_reach_paddle(self, ball: Ball) -> float:
        """Estimated ticks until the ball reaches the left paddle's x position. Uses time = distance / speed to estimate the number of ticks until
        hit."""

        paddle_x = self.game.left_paddle.position_x
        right_wall_x = self.grid_size_x - 1
        velocity_x = ball.velocity_x

        if velocity_x < 0:
            if ball.position_x < paddle_x:
                return math.inf  # already behind the paddle, cannot be saved so least relevant
            return (ball.position_x - paddle_x) / abs(velocity_x)
        if velocity_x > 0:
            return ((right_wall_x - ball.position_x) + (right_wall_x - paddle_x)) / abs(velocity_x)
        return math.inf  # no horizontal motion, never arrives

    def _get_obs(self) -> ObservationT:
        """Returns an observation"""

        balls_by_proximity = sorted(self.game.balls, key=self._time_to_reach_paddle)

        # The observation shape is [{paddle observation}, {closest_ball_observation}, {next_closest}, ....]
        observation_list = []

        # Paddle observation is: [position_y, position_x]
        observation_list.extend([self.game.left_paddle.position_y / self.grid_size_y, self.game.left_paddle.position_x / self.grid_size_x])

        # Each balls observation is in format: [present, ball_position_y, ball_position_x, ball_velocity_y, ball_velocity_x]
        for _ in range(MAX_BALLS):
            if len(balls_by_proximity):
                ball = balls_by_proximity.pop(0)  # take the first one off the list, this is closest to the paddle.
                # Normalise these values here. They should always be between -1 and 1.
                present = 1
                ball_position_y = ball.position_y / self.grid_size_y
                ball_position_x = ball.position_x / self.grid_size_x
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

    def _get_info(self) -> Dict[str, Any]:
        """Returns nothing for the time being but can be used to return debugging information"""
        return {"test": "testing"}

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[ObservationT, Dict[str, Any]]:
        """Resets the environment"""

        super().reset(seed=seed)
        self.game = Game(self.np_random, single_player=True, logger=None)  # reset the game state

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def _calculate_reward(self, tick_result: Dict[Player, List[TickEvent]]) -> Dict[Player, int]:
        """Calculates the reward for the given step."""

        player_scores = {}
        for player, events in tick_result.items():
            score = 0
            for event in events:
                if event == TickEvent.LIFE_LOST:
                    score -= 10
                elif event == TickEvent.PADDLE_BOUNCE:
                    if self.game.level == Level.EASY:
                        score += 1
                    elif self.game.level == Level.MEDIUM:
                        score += 3
                    elif self.game.level == Level.HARD:
                        score += 6
                    else:
                        raise ValueError(f"Unhandled level {self.game.level}")

            player_scores[player] = score

        return player_scores

    def step(self, action: ActionT) -> Tuple[ObservationT, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Steps the game forwards one"""

        converted_action = PaddleMove(action - 1)  # action space starts at 0, but our PaddleMove IntEnum starts at -1, so need to shift -1.
        tick_result = self.game.tick({Player.LEFT: converted_action})
        player_rewards = self._calculate_reward(tick_result)

        # TODO: only supports a single player reward
        left_player_reward = player_rewards[Player.LEFT]

        terminated = self.game.left_lives <= 0
        if not terminated:
            assert self.game.winner is None

        truncated = False

        observation = self._get_obs()
        info = self._get_info()
        return observation, left_player_reward, terminated, truncated, info

check_env(TrainingGame())