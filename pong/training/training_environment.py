"""Used for training the RL model"""

from typing import Any, Dict, Tuple, Optional, SupportsFloat, List

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import numpy as np

from pong.game.game_constants import GRID_SIZE_Y, GRID_SIZE_X, MAX_BALLS, RANDOM_SEED
from pong.game.game import Game, Player, PaddleMove, TickEvent, Level
from pong.training.model_common import ModelObservationT, ModelActionT, get_observation

MAX_PADDLE_BOUNCES = 1000  # truncate after 1000 paddle bounces

LIFE_LOST_REWARD = -10
LEVEL_EASY_REWARD = 1
LEVEL_MEDIUM_REWARD = 2
LEVEL_HARD_REWARD = 3


class PongTrainingEnvironment(gym.Env):
    def __init__(self) -> None:
        super().__init__(seed=RANDOM_SEED)

        self.grid_size_y = GRID_SIZE_Y
        self.grid_size_x = GRID_SIZE_X

        shape = 2 + (MAX_BALLS * 5)
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(shape,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3, start=0, dtype=np.int8)

        self.game = Game(self.np_random, single_player=True, logger=None)  # Start with a fresh game

    def _get_info(self) -> Dict[str, Any]:
        """Returns nothing for the time being but can be used to return debugging information"""
        return {"test": "testing"}

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[ModelObservationT, Dict[str, Any]]:
        """Resets the environment"""

        super().reset(seed=seed)
        self.game = Game(self.np_random, single_player=True, logger=None)  # reset the game state

        observation = get_observation(self.game)
        info = self._get_info()
        return observation, info

    def _calculate_reward(self, tick_result: Dict[Player, List[TickEvent]]) -> Dict[Player, int]:
        """Calculates the reward for the given step."""

        player_scores = {}
        for player, events in tick_result.items():
            score = 0
            for event in events:
                if event == TickEvent.LIFE_LOST:
                    score -= LIFE_LOST_REWARD
                elif event == TickEvent.PADDLE_BOUNCE:
                    if self.game.level == Level.EASY:
                        score += LEVEL_EASY_REWARD
                    elif self.game.level == Level.MEDIUM:
                        score += LEVEL_MEDIUM_REWARD
                    elif self.game.level == Level.HARD:
                        score += LEVEL_HARD_REWARD
                    else:
                        raise ValueError(f"Unhandled level {self.game.level}")

            player_scores[player] = score

        return player_scores

    def step(self, action: ModelActionT) -> Tuple[ModelObservationT, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Steps the game forwards one"""

        converted_action = PaddleMove(int(action) - 1)  # action space starts at 0, but our PaddleMove IntEnum starts at -1, so need to shift -1.
        tick_result = self.game.tick({Player.LEFT: converted_action})
        player_rewards = self._calculate_reward(tick_result)

        # TODO: only supports a single player reward
        left_player_reward = player_rewards[Player.LEFT]

        terminated = self.game.left_lives <= 0
        if not terminated:
            assert self.game.winner is None

        truncated = self.game.paddle_bounces >= MAX_PADDLE_BOUNCES

        observation = get_observation(self.game)
        info = self._get_info()
        return observation, left_player_reward, terminated, truncated, info


check_env(PongTrainingEnvironment())
