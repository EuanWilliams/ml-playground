"""Used for training the RL model"""

from typing import Any, Dict, Tuple, Optional, SupportsFloat, List

import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

from pong.controllers import AutoController
from pong.game.game_constants import GRID_SIZE_Y, GRID_SIZE_X, MAX_BALLS
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

MAX_FAST_FORWARD_TICKS = 100_000
MAX_START_PADDLE_BOUNCES = 50


class PongTrainingEnvironment(gym.Env):
    def __init__(self, random_start: bool = True) -> None:
        super().__init__()

        self.grid_size_y = GRID_SIZE_Y
        self.grid_size_x = GRID_SIZE_X

        self.random_start = random_start

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
        if self.random_start:
            target_bounces = int(self.np_random.integers(0, MAX_START_PADDLE_BOUNCES + 1))
            starting_lives = self.game.left_lives
            auto_controller = AutoController(None)  # use the auto controller to fast forward the game to a random point in the game
            for _ in range(MAX_FAST_FORWARD_TICKS):
                if self.game.paddle_bounces >= target_bounces:
                    break

                auto_controller.read_observation(self.game, None)
                action = auto_controller.get_move()
                self.game.tick({Player.LEFT: action})
                self.game.left_lives = starting_lives  # keep the scripted player alive so it always reaches the target

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
