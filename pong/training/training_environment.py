"""Used for training the RL model"""
import math
from typing import Any, Dict, Tuple, Optional, SupportsFloat, List

import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

from pong.game.game_constants import GRID_SIZE_Y, GRID_SIZE_X, MAX_BALLS, LEVEL_EASY_MAX_BOUNCES, LEVEL_MEDIUM_MAX_BOUNCES, LEVEL_HARD_NEW_BALLS_PADDLE_BOUNCES
from pong.game.ball import Ball
from pong.game.game import Game, Player, PaddleMove, TickEvent, Level
from pong.training.model_common import ModelObservationT, ModelActionT, get_observation

MAX_PADDLE_BOUNCES = 1000

LEVEL_EASY_LIFE_LOST_REWARD = -20
LEVEL_MEDIUM_LIFE_LOST_REWARD = -20
LEVEL_HARD_LIFE_LOST_REWARD = -20

LEVEL_PADDLE_BOUNCE_EASY_REWARD = 10
LEVEL_PADDLE_BOUNCE_MEDIUM_REWARD = 10
LEVEL_PADDLE_BOUNCE_HARD_REWARD = 10

NO_EVENT_REWARD = 1

EVENT_TO_GAME_LEVEL_REWARD = {
    TickEvent.LIFE_LOST: {
        Level.EASY: LEVEL_EASY_LIFE_LOST_REWARD,
        Level.MEDIUM: LEVEL_MEDIUM_LIFE_LOST_REWARD,
        Level.HARD: LEVEL_HARD_LIFE_LOST_REWARD,
    },
    TickEvent.PADDLE_BOUNCE: {
        Level.EASY: LEVEL_PADDLE_BOUNCE_EASY_REWARD,
        Level.MEDIUM: LEVEL_PADDLE_BOUNCE_MEDIUM_REWARD,
        Level.HARD: LEVEL_PADDLE_BOUNCE_HARD_REWARD,
    },
}


class PongTrainingEnvironment(gym.Env):
    def __init__(self) -> None:
        super().__init__()

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

        # when we reset, instead of starting from scratch, randomly start at a point in the game to give the model more variety and exposure to later game
        # phases. The game uses paddle bounces to determine level. Currently the game uses the following level progression:
        # Level EASY: 0-9 bounces - normal acceleration
        # Level MEDIUM: 10-19 bounces - 2x normal acceleration
        # Level HARD: 20+ bounces - 2x normal acceleration + a new ball every 10 bounces.
        starting_paddle_bounces = self.np_random.integers(0, 20)
        self.game.paddle_bounces = int(starting_paddle_bounces)
        if starting_paddle_bounces < LEVEL_EASY_MAX_BOUNCES:
            self.game.level = Level.EASY
        elif starting_paddle_bounces < LEVEL_MEDIUM_MAX_BOUNCES:
            self.game.level = Level.MEDIUM
        else:
            self.game.level = Level.HARD
            if starting_paddle_bounces >= LEVEL_MEDIUM_MAX_BOUNCES + LEVEL_HARD_NEW_BALLS_PADDLE_BOUNCES:
                balls_required = math.floor((starting_paddle_bounces - LEVEL_MEDIUM_MAX_BOUNCES) / LEVEL_HARD_NEW_BALLS_PADDLE_BOUNCES)
                if balls_required > 0:
                    self.game.balls.extend([Ball(self.np_random) for _ in range(balls_required)])
                    self.game.last_new_ball_paddle_bounce = math.floor(starting_paddle_bounces / LEVEL_HARD_NEW_BALLS_PADDLE_BOUNCES) * LEVEL_HARD_NEW_BALLS_PADDLE_BOUNCES

        observation = get_observation(self.game)
        info = self._get_info()
        return observation, info

    def _calculate_reward(self, tick_result: Dict[Player, List[TickEvent]]) -> Dict[Player, int]:
        """Calculates the reward for the given step."""

        player_scores = {}
        for player, events in tick_result.items():
            score = 0
            if len(events) == 0:
                score += NO_EVENT_REWARD
            else:
                for event in events:
                    score += EVENT_TO_GAME_LEVEL_REWARD[event][self.game.level]

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
