from enum import IntEnum
from typing import List, Optional, Dict

import numpy as np

from pong.game.ball import Ball
from pong.game.paddle import Paddle
from pong.game.game_constants import (
    DEATH_ZONE_LEFT,
    DEATH_ZONE_RIGHT,
    FLOOR,
    CEILING,
    LEFT_PADDLE,
    RIGHT_PADDLE,
    BOUNCING_OBJECTS,
    BOUNCING_OBJECTS_SINGLE_PLAYER,
    DEATH_ZONES,
    DEATH_ZONES_SINGLE_PLAYER,
    STARTING_LIVES,
    LEVEL_EASY_MAX_BOUNCES,
    LEVEL_MEDIUM_MAX_BOUNCES,
    LEVEL_EASY_SCORE_POINTS,
    LEVEL_MEDIUM_SCORE_POINTS,
    LEVEL_HARD_SCORE_POINTS,
    GRID_SIZE_Y,
    GRID_SIZE_X,
    LEVEL_EASY_BALL_Y_ACCELERATION,
    LEVEL_EASY_BALL_X_ACCELERATION,
    LEVEL_MEDIUM_HIGH_BALL_Y_ACCELERATION,
    LEVEL_MEDIUM_HIGH_BALL_X_ACCELERATION,
    OBJECT_TO_UNIT_NORMAL_VECTOR,
    LEFT_PADDLE_X_VALUE,
    RIGHT_PADDLE_X_VALUE,
    MAX_BALLS,
    LEVEL_HARD_NEW_BALLS_PADDLE_BOUNCES,
)


class Level(IntEnum):
    EASY = 0
    MEDIUM = 1
    HARD = 2


class Player(IntEnum):
    LEFT = 0
    RIGHT = 1


class PaddleMove(IntEnum):
    UP = -1
    STAY = 0
    DOWN = 1


class PlayerAction(IntEnum):
    LEFT_PADDLE_UP = 0
    LEFT_PADDLE_DOWN = 1
    RIGHT_PADDLE_UP = 2
    RIGHT_PADDLE_DOWN = 3


class TickEvent(IntEnum):
    """Events that can occur during a tick."""

    PADDLE_BOUNCE = 0
    LIFE_LOST = 1


class GameLogger:
    """Simple game logger for storing and emitting log messages at the end of game if needed."""

    def __init__(self) -> None:
        self.log_messages: List[str] = []

    def add_log_message(self, message: str) -> None:
        """Add a log message to list"""
        self.log_messages.append(message)

    def output_log_messages(self) -> None:
        """Emits log messages"""

        for log_message in self.log_messages:
            print(log_message)


class Game:
    """The game"""

    def __init__(self, random: np.random.Generator, single_player: bool, logger: Optional[GameLogger]) -> None:
        self.random = random
        self.logger = logger
        self.single_player = single_player

        self._init_new_game()

    def _init_new_game(self) -> None:
        """Initialises a new game"""

        self.level: Level = Level.EASY

        self.balls = [Ball(self.random)]  # start with a single ball.
        paddle_starting_y = GRID_SIZE_Y // 2
        self.left_paddle = Paddle(starting_y=paddle_starting_y, x_position=LEFT_PADDLE_X_VALUE)
        self.right_paddle = Paddle(starting_y=paddle_starting_y, x_position=RIGHT_PADDLE_X_VALUE) if not self.single_player else None

        self.left_lives = STARTING_LIVES
        self.right_lives = STARTING_LIVES if not self.single_player else None
        self.last_life_lost_tick: Optional[int] = None  # set to tick_index when a player loses a life for display in TUI
        self.winner: Optional[Player] = None
        self.paddle_bounces: int = 0
        self.score = 0

        self.acceleration_y = LEVEL_EASY_BALL_Y_ACCELERATION
        self.acceleration_x = LEVEL_EASY_BALL_X_ACCELERATION
        self.bouncing_objects = BOUNCING_OBJECTS if not self.single_player else BOUNCING_OBJECTS_SINGLE_PLAYER
        self.death_zone_objects = DEATH_ZONES if not self.single_player else DEATH_ZONES_SINGLE_PLAYER

        self.tick_index = 0
        self.last_new_ball_paddle_bounce = (
            0  # set to the paddle bounce that the last new ball was added at, used to avoid adding new balls between paddle bounces
        )
        self.tick_events: Dict[Player, List[TickEvent]] = {}  # reset every tick.

    def _log(self, message: str) -> None:
        """Appends log message"""

        if self.logger is not None:
            self.logger.add_log_message(f"{self.tick_index}: {message}")

    def _register_player_move(self, paddle_move: PaddleMove, player: Player) -> None:
        """Responds to a player move by updating the paddle state"""

        self._log(f"Registering player move: {paddle_move.name}")

        if player == Player.RIGHT and self.single_player:
            raise ValueError(f"Trying to move right paddle but in single player mode, this should not happen.")

        paddle = self.left_paddle if player == Player.LEFT else self.right_paddle
        assert paddle is not None

        if paddle_move == PaddleMove.UP:
            paddle.move_paddle(-1)
        elif paddle_move == PaddleMove.DOWN:
            paddle.move_paddle(1)
        elif paddle_move != PaddleMove.STAY:
            raise ValueError(f"Unhandled player move: {paddle_move}")

    def _check_collision(self, ball: Ball) -> Optional[int]:
        """Checks if the next ball position will collide with an object"""

        next_position_y, next_position_x = ball.next_position

        if next_position_y <= 0:
            return CEILING
        elif next_position_y >= GRID_SIZE_Y - 1:
            return FLOOR
        elif next_position_x <= 0:
            # Tick events for deaths are handled in the life lost function.
            return DEATH_ZONE_LEFT
        elif next_position_x >= GRID_SIZE_X - 1:
            # Tick events for deaths are handled in the life lost function.
            return DEATH_ZONE_RIGHT
        elif self.left_paddle.hits_paddle(next_position_y, next_position_x):
            self.tick_events.setdefault(Player.LEFT, []).append(TickEvent.PADDLE_BOUNCE)
            return LEFT_PADDLE
        elif not self.single_player and self.right_paddle is not None and self.right_paddle.hits_paddle(next_position_y, next_position_x):
            self.tick_events.setdefault(Player.RIGHT, []).append(TickEvent.PADDLE_BOUNCE)
            return RIGHT_PADDLE
        else:
            return None

    def _handle_life_lost(self, ball: Ball, death_zone: int) -> None:
        """Handles a player losing a life"""

        self._log(f"Life lost to death zone {death_zone}")

        if death_zone == DEATH_ZONE_LEFT:
            self.left_lives -= 1
            self.tick_events.setdefault(Player.LEFT, []).append(TickEvent.LIFE_LOST)
        elif death_zone == DEATH_ZONE_RIGHT:
            assert self.right_lives is not None
            assert not self.single_player
            self.right_lives -= 1
            self.tick_events.setdefault(Player.RIGHT, []).append(TickEvent.LIFE_LOST)
        else:
            raise ValueError(f"Invalid object for losing a life: {death_zone}")

        if self.left_lives <= 0:
            self.winner = Player.RIGHT
        elif self.right_lives is not None and self.right_lives <= 0:
            self.winner = Player.LEFT
        else:
            ball.reset_to_start()

        self.last_life_lost_tick = self.tick_index
        self._log(f"Life lost to death zone, new scores: Left {self.left_lives} : {self.right_lives} Right")

    def _handle_ball_bounce(self, ball: Ball) -> None:
        """Handles a ball bouncing against an object"""

        ball.accelerate(self.acceleration_y, self.acceleration_x)

        self.paddle_bounces += 1

        if self.level == Level.EASY:
            self.score += LEVEL_EASY_SCORE_POINTS
        elif self.level == Level.MEDIUM:
            self.score += LEVEL_MEDIUM_SCORE_POINTS
        elif self.level == Level.HARD:
            self.score += LEVEL_HARD_SCORE_POINTS
        else:
            raise ValueError(f"Unhandled level {self.level.name}")

    def _update_level(self) -> None:
        """Checks if its time to update the level."""

        if self.paddle_bounces < LEVEL_EASY_MAX_BOUNCES:
            if self.level != Level.EASY:
                raise ValueError(f"Unxpected level with paddle bounces {self.paddle_bounces}, level: {self.level}")
        elif self.paddle_bounces < LEVEL_MEDIUM_MAX_BOUNCES:
            if self.level == Level.EASY:
                # in level easy moving to level medium, increase speed.
                self.level = Level.MEDIUM
                self.acceleration_y = LEVEL_MEDIUM_HIGH_BALL_Y_ACCELERATION
                self.acceleration_x = LEVEL_MEDIUM_HIGH_BALL_X_ACCELERATION
            elif self.level != Level.MEDIUM:
                raise ValueError(f"Unexpected level with paddle bounces {self.paddle_bounces}, level: {self.level}")
        else:
            if self.level == Level.MEDIUM:
                # in level medium moving to hard, add another ball.
                self.level = Level.HARD
                self.balls.append(Ball(self.random))
                self.last_new_ball_paddle_bounce = self.paddle_bounces
            elif self.level == Level.HARD:
                # While in level HARD, keep adding new balls until we get to max balls.
                if self.paddle_bounces >= self.last_new_ball_paddle_bounce + LEVEL_HARD_NEW_BALLS_PADDLE_BOUNCES and len(self.balls) < MAX_BALLS:
                    self.balls.append(Ball(self.random))
                    self.last_new_ball_paddle_bounce = self.paddle_bounces
            else:
                raise ValueError(f"Unexpected level with paddle bounces {self.paddle_bounces}, level: {self.level}")

    def tick(self, actions: Dict[Player, PaddleMove]) -> Dict[Player, List[TickEvent]]:
        """Ticks the game's state by one increment"""

        self._log("Starting tick")
        self.tick_events = {Player.LEFT: []}  # must be reset every tick.
        if not self.single_player:
            self.tick_events[Player.RIGHT] = []

        # Handle the player's action
        if (left_player_action := actions.get(Player.LEFT)) is not None:
            self._register_player_move(left_player_action, Player.LEFT)
        if (right_player_action := actions.get(Player.RIGHT)) is not None:
            self._register_player_move(right_player_action, Player.RIGHT)

        # Update ball position
        for ball in self.balls:
            collision_value = self._check_collision(ball)
            if collision_value in self.bouncing_objects:
                ball.bounce(OBJECT_TO_UNIT_NORMAL_VECTOR[collision_value])
                if collision_value in [LEFT_PADDLE, RIGHT_PADDLE]:
                    self._handle_ball_bounce(ball)
            elif collision_value in self.death_zone_objects:
                assert collision_value is not None
                self._handle_life_lost(ball, collision_value)
            else:
                self._log("No collision, ticking ball")
                ball.tick()

        # update level
        self._update_level()
        self.tick_index += 1

        return self.tick_events
