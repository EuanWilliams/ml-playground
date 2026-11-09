"""Used for training the RL model"""

from termios import VLNEXT
from typing import Any, Dict, Tuple, Optional, SupportsFloat, List

import gymnasium as gym
from gymnasium.core import ObsType

from pong.game import Game, GameLogger, Player, PaddleMove, TickEvent, Level

TICK_TRUNCATE_VALUE = 100  # Stop if the game gets this far


class TrainingGame(Game, gym.Env):
    def __init__(self, logger: GameLogger, grid_y: int, grid_x: int) -> None:
        super().__init__(logger, single_player=True)

        self.grid_size_y = grid_y
        self.grid_size_x = grid_x

        self.observation_space = gym.spaces.Dict(
            {
                "ball_position_y": gym.spaces.Box(low=1, high=self.grid_size_y - 1, dtype=float),
                "ball_position_x": gym.spaces.Box(low=1, high=self.grid_size_x - 1, dtype=float),
                "ball_velocity_y": gym.spaces.Box(low=0, dtype=float),
                "paddle_position_y": gym.spaces.Box(low=1, high=self.grid_size_y - 1, dtype=int),
            }
        )

        self.action_space = gym.spaces.Discrete(3, start=-1, dtype=int)

    def _get_obs(self):
        """Returns an observation"""
        # TODO - typing

        return {
            "ball_position_y": self.balls[0].position_y,
            "ball_position_x": self.balls[0].position_x,
            "ball_velocity_y": self.balls[0].velocity_y,
            "paddle_position_y": self.left_paddle.position_y,
        }

    def _get_info(self) -> Dict[str, Any]:
        """Returns nothing for the time being but can be used to return debugging information"""
        return {"test": "testing"}

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[ObsType, Dict[str, Any]]:
        """Resets the environment"""

        super().reset(seed=seed)

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def _calculate_reward(self, tick_result: Dict[Player, List[TickEvent]]) -> Dict[Player, float]:
        """Calculates the reward for the given step."""

        player_scores = {}
        for player, events in tick_result.items():
            score = 0
            for event in events:
                if event == TickEvent.LIFE_LOST:
                    score -= 10
                elif event == TickEvent.PADDLE_BOUNCE:
                    if self.level == Level.EASY:
                        score += 1
                    elif self.level == Level.MEDIUM:
                        score += 3
                    elif self.level == Level.HARD:
                        score += 6
                    else:
                        raise ValueError(f"Unhandled level {self.level}")
                else:
                    raise ValueError(f"Unhandled tick event: {event}")

            player_scores[player] = score

        return player_scores

    def step(self, action) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Steps the game forwards one"""

        tick_result = self.tick({Player.LEFT: PaddleMove(action)})
        player_rewards = self._calculate_reward(tick_result)

        # TODO: only supports a single player reward
        left_player_reward = player_rewards[Player.LEFT]

        terminated = self.left_lives == 0
        truncated = False

        observation = self._get_obs()
        info = self._get_info()
        return observation, left_player_reward, terminated, truncated, info
