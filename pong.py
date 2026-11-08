import curses
import math
import time
import random
from enum import IntEnum
from typing import List, Tuple, Optional

import numpy as np
from numpy import typing as npt

Grid = npt.NDArray[np.float32]


class Player(IntEnum):
    LEFT = 0
    RIGHT = 1


class PlayerAction(IntEnum):
    LEFT_PADDLE_UP = 0
    LEFT_PADDLE_DOWN = 1
    RIGHT_PADDLE_UP = 2
    RIGHT_PADDLE_DOWN = 3


# Game constants
EMPTY_SPACE = 0
DEATH_ZONE_LEFT = 1
DEATH_ZONE_RIGHT = 2
FLOOR = 3
CEILING = 4
LEFT_PADDLE = 5
RIGHT_PADDLE = 6
BALL = 9

BOUNCING_OBJECTS = [FLOOR, CEILING, LEFT_PADDLE, RIGHT_PADDLE]
BOUNCING_OBJECTS_SINGLE_PLAYER = BOUNCING_OBJECTS + [DEATH_ZONE_RIGHT]
DEATH_ZONES = [DEATH_ZONE_LEFT, DEATH_ZONE_RIGHT]
DEATH_ZONES_SINGLE_PLAYER = [DEATH_ZONE_LEFT]

STARTING_LIVES = 3

# TUI Game
TICK_SLEEP_TIME = 1 / 30

DISPLAY_CHARACTERS = {
    EMPTY_SPACE: " ",
    DEATH_ZONE_LEFT: "|",
    DEATH_ZONE_RIGHT: "|",
    CEILING: "-",
    FLOOR: "-",
    BALL: "0",
    LEFT_PADDLE: "[",
    RIGHT_PADDLE: "]",
}

KEY_W = 119
KEY_S = 115
KEY_UP_ARROW = 259
KEY_DOWN_ARROW = 258

KEYSTROKE_TO_ACTION = {
    KEY_W: PlayerAction.LEFT_PADDLE_UP,
    KEY_S: PlayerAction.LEFT_PADDLE_DOWN,
    KEY_UP_ARROW: PlayerAction.RIGHT_PADDLE_UP,
    KEY_DOWN_ARROW: PlayerAction.RIGHT_PADDLE_DOWN,
}

# Physical constants
GRID_SIZE_Y = 10
GRID_SIZE_X = 30

LEFT_PADDLE_X_VALUE = 1
RIGHT_PADDLE_X_VALUE = GRID_SIZE_X - 2

STARTING_Y_VELOCITY = 0.1
STARTING_X_VELOCITY = 0.3

BALL_X_ACCELERATION = 0.025  # ball accelerates this amount per bounce on a paddle
BALL_Y_ACCELERATION = 0.0625

# Maps object to unit normal vector, used in the calculation of velocity after a bounce.
OBJECT_TO_UNIT_NORMAL_VECTOR = {
    DEATH_ZONE_LEFT: np.array((0, 1)),  # Vertical wall (left)
    DEATH_ZONE_RIGHT: np.array((0, -1)),  # Vertical wall (right)
    FLOOR: np.array((1, 0)),  # Horizontal floor
    CEILING: np.array((-1, 0)),  # Horizontal ceiling
    LEFT_PADDLE: np.array((0, 1)),
    RIGHT_PADDLE: np.array((0, -1)),
}


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


class Paddle:
    """A paddle"""

    def __init__(self, player: Player, starting_y: int) -> None:
        self.player = player
        self.position_y = starting_y

    @property
    def position_x(self) -> int:
        """Returns the x position value based on side"""
        return RIGHT_PADDLE_X_VALUE if self._get_side_safe() == Player.RIGHT else LEFT_PADDLE_X_VALUE

    @property
    def position(self) -> Tuple[int, int]:
        """Returns the paddle position in format y, x"""
        return self.position_y, self.position_x

    @property
    def display_character(self) -> str:
        """Returns the display character to render"""
        return DISPLAY_CHARACTERS[RIGHT_PADDLE] if self._get_side_safe() == Player.RIGHT else DISPLAY_CHARACTERS[LEFT_PADDLE]

    def _get_side_safe(self) -> Player:
        """Returns the side after validating"""

        if self.player not in [Player.LEFT, Player.RIGHT]:
            raise ValueError(f"Invalid side: {self.player}")

        return self.player

    def move_paddle(self, increment: int) -> None:
        """Moves the ball a single increment"""
        self.position_y = max(min(self.position_y + increment, GRID_SIZE_Y - 2), 1)

    def hits_paddle(self, ball_position_y: int, ball_position_x: int) -> bool:
        """Checks if the ball collides with the paddle"""
        return ball_position_y == self.position_y and ball_position_x == self.position_x


class Ball:
    """The ball"""

    display_character = DISPLAY_CHARACTERS[BALL]

    def __init__(self) -> None:
        self.position_x = 0.0
        self.position_y = 0.0

        self.velocity_y = 0.0
        self.velocity_x = 0.0
        self.reset_to_start()

    @property
    def position(self) -> Tuple[float, float]:
        """Returns the balls position"""
        return self.position_y, self.position_x

    @property
    def rounded_position(self) -> Tuple[int, int]:
        """Returns the balls position"""
        return round(self.position_y), round(self.position_x)

    @property
    def next_position(self) -> Tuple[int, int]:
        """Returns the ball's next position, without actually moving the ball, snapped to an integer."""
        return round(self.position_y + self.velocity_y), round(self.position_x + self.velocity_x)

    def tick(self) -> None:
        """Increments the balls position by one tick"""
        self.position_y = self.position_y + self.velocity_y
        self.position_x = self.position_x + self.velocity_x

    def bounce(self, unit_normal_vector: npt.NDArray[np.uint8]) -> None:
        """Bounces the ball, calculates the resulting velocity of the ball after the bounce.
        Calculates resulting velocity using the equation:

                vout = vin - 2(vin . N)N

        Where N is the unit normal vector of the surface.
        """

        # Calculate new velocity
        input_velocity = np.array([self.velocity_y, self.velocity_x], dtype=np.float32)
        self.velocity_y, self.velocity_x = input_velocity - (2 * np.dot(input_velocity, unit_normal_vector) * unit_normal_vector)

        # Send the ball off in the new velocity
        self.position_y = self.position_y + self.velocity_y
        self.position_x = self.position_x + self.velocity_x

    def accelerate(self, delta_y: float, delta_x: float) -> None:
        """Accelerates the ball a provided amount"""

        self.velocity_y += math.copysign(delta_y, self.velocity_y)
        self.velocity_x += math.copysign(delta_x, self.velocity_x)

    def set_position(self, position: Tuple[int, int]) -> None:
        """Sets the balls position"""
        self.position_y, self.position_x = position

    def reset_to_start(self) -> None:
        """Sets the ball to the middle of the frame and randomises starting velocity"""

        # y velocity in positions per tick - positive means moving top to bottom
        self.velocity_y = STARTING_Y_VELOCITY * random.choice([1, -1])
        # x velocity in positions per tick - positive means moving left to right
        self.velocity_x = STARTING_X_VELOCITY * random.choice([1, -1])

        self.position_y = GRID_SIZE_Y // 2
        self.position_x = GRID_SIZE_X // 2


class Game:
    """The game"""

    def __init__(self, logger: GameLogger, single_player: bool) -> None:
        self.logger = logger
        self.single_player = single_player

        self.ball = Ball()
        paddle_starting_y = GRID_SIZE_Y // 2
        self.left_paddle = Paddle(Player.LEFT, starting_y=paddle_starting_y)
        self.right_paddle = Paddle(Player.RIGHT, starting_y=paddle_starting_y) if not self.single_player else None

        self.left_lives = STARTING_LIVES
        self.right_lives = STARTING_LIVES if not self.single_player else None
        self.last_life_lost_tick: Optional[int] = None  # set to tick_index when a player loses a life for display in TUI
        self.winner: Optional[Player] = None
        self.paddle_bounces: int = 0

        self.bouncing_objects = BOUNCING_OBJECTS if not self.single_player else BOUNCING_OBJECTS_SINGLE_PLAYER
        self.death_zone_objects = DEATH_ZONES if not self.single_player else DEATH_ZONES_SINGLE_PLAYER

        self.tick_index = 0
        self.log_messages: List[str] = []

    def _log(self, message: str) -> None:
        """Appends log message"""
        self.logger.add_log_message(f"{self.tick_index}: {message}")

    def _register_player_action(self, player_action: PlayerAction) -> None:
        """Responds to a player action by updating the paddle state"""

        self._log(f"Registering player action: {player_action.name}")

        if player_action == PlayerAction.LEFT_PADDLE_UP:
            self.left_paddle.move_paddle(-1)
        elif player_action == PlayerAction.LEFT_PADDLE_DOWN:
            self.left_paddle.move_paddle(1)
        elif player_action == PlayerAction.RIGHT_PADDLE_UP:
            if self.right_paddle is not None:
                self.right_paddle.move_paddle(-1)
            else:
                assert self.single_player, f"Right paddle is none but single player is false, this should not happen."
                self._log(f"Ignoring right paddle player action in single player mode.")
        elif player_action == PlayerAction.RIGHT_PADDLE_DOWN:
            if self.right_paddle is not None:
                self.right_paddle.move_paddle(1)
            else:
                assert self.single_player, f"Right paddle is none but single player is false, this should not happen."
                self._log(f"Ignoring right paddle player action in single player mode.")
        else:
            raise ValueError(f"Unhandled player action: {player_action}")

    def _check_collision(self) -> Optional[int]:
        """Checks if the next ball position will collide with an object"""

        next_position_y, next_position_x = self.ball.next_position

        if next_position_y <= 0:
            return CEILING
        elif next_position_y >= GRID_SIZE_Y - 1:
            return FLOOR
        elif next_position_x <= 0:
            return DEATH_ZONE_LEFT
        elif next_position_x >= GRID_SIZE_X - 1:
            return DEATH_ZONE_RIGHT
        elif self.left_paddle.hits_paddle(next_position_y, next_position_x):
            return LEFT_PADDLE
        elif not self.single_player and self.right_paddle is not None and self.right_paddle.hits_paddle(next_position_y, next_position_x):
            return RIGHT_PADDLE
        else:
            return None

    def _handle_life_lost(self, death_zone: int) -> None:
        """Handles a player losing a life"""

        self._log(f"Life lost to death zone {death_zone}")

        if death_zone == DEATH_ZONE_LEFT:
            self.left_lives -= 1
        elif death_zone == DEATH_ZONE_RIGHT:
            assert self.right_lives is not None
            self.right_lives -= 1
        else:
            raise ValueError(f"Invalid object for losing a life: {death_zone}")

        if self.left_lives == 0:
            self.winner = Player.RIGHT
        elif self.right_lives == 0:
            self.winner = Player.LEFT
        else:
            self.ball.reset_to_start()

        self.last_life_lost_tick = self.tick_index
        self._log(f"Life lost to death zone, new scores: Left {self.left_lives} : {self.right_lives} Right")

    def tick(self, player_action: Optional[PlayerAction]) -> None:
        """Ticks the game's state by one increment"""

        self._log("Starting tick")

        # Handle the player's action
        if player_action is not None:
            self._register_player_action(player_action)

        # Update ball position
        collision_value = self._check_collision()
        if collision_value in BOUNCING_OBJECTS:
            self.ball.bounce(OBJECT_TO_UNIT_NORMAL_VECTOR[collision_value])
            if collision_value in [LEFT_PADDLE, RIGHT_PADDLE]:
                self.ball.accelerate(BALL_Y_ACCELERATION, BALL_X_ACCELERATION)
                self.paddle_bounces += 1
        elif collision_value in [DEATH_ZONE_LEFT, DEATH_ZONE_RIGHT]:
            self._handle_life_lost(collision_value)
        else:
            self._log("No collision, ticking ball")
            self.ball.tick()

        self.tick_index += 1


class TerminalApp:
    """Terminal app for running pong interactively in the terminal"""

    def __init__(self, stdscr: curses.window, game: Game, logger: GameLogger) -> None:
        self.stdscr = stdscr
        self.game = game
        self.logger = logger
        self.single_player = self.game.single_player

        self.grid = self._init_grid()

    def _game_over(self) -> None:
        """Splashes a screen to say game over."""
        if self.single_player:
            lines = [
                "GAME OVER",
                "",
                f"Final score: {self.game.paddle_bounces}",
                "",
                "Press any key",
            ]
        else:
            winner, loser = ("Left", "Right") if self.game.right_lives == 0 else ("Right", "Left")
            lines = [
                "GAME OVER",
                "",
                f"{winner} wins.",
                f"{loser} loses.",
                "",
                f"Left {self.game.left_lives} : {self.game.right_lives} Right",
                "",
                "Press any key",
            ]

        window = curses.newwin(GRID_SIZE_Y, GRID_SIZE_X, 0, 0)
        window.box()
        for row, line in enumerate(lines):
            window.addstr(row + 1, (GRID_SIZE_X - len(line)) // 2, line)

        self.stdscr.clear()
        self.stdscr.refresh()
        window.refresh()

        self.stdscr.nodelay(False)
        self.stdscr.getch()

    def _render(self) -> None:
        """Renders the current status to terminal"""

        # overlay grid
        frame = "\n".join("".join(DISPLAY_CHARACTERS[cell] for cell in row) for row in self.grid)
        self.stdscr.addstr(0, 0, frame)

        # overlay paddles
        self.stdscr.addstr(self.game.left_paddle.position[0], self.game.left_paddle.position[1], self.game.left_paddle.display_character)
        if not self.single_player:
            assert self.game.right_paddle is not None
            self.stdscr.addstr(self.game.right_paddle.position[0], self.game.right_paddle.position[1], self.game.right_paddle.display_character)

        # overlay ball
        self.stdscr.addstr(self.game.ball.rounded_position[0], self.game.ball.rounded_position[1], self.game.ball.display_character)

        # overlay bottom line
        if self.game.last_life_lost_tick is not None and self.game.tick_index - self.game.last_life_lost_tick <= 25:
            bottom_line = f"Life lost (×_×)".center(GRID_SIZE_X)
        elif self.single_player:
            bottom_line = f"Current score: {self.game.paddle_bounces} | Lives: {self.game.left_lives}".center(GRID_SIZE_X)
        else:
            bottom_line = f"Left {self.game.left_lives} : {self.game.right_lives} Right".center(GRID_SIZE_X)

        self.stdscr.addstr(len(self.grid), 0, bottom_line)
        self.stdscr.refresh()

    def run(self) -> None:
        """Runs the game, render status to terminal"""

        try:
            self.stdscr.nodelay(True)

            while self.game.winner is None:
                time.sleep(TICK_SLEEP_TIME)

                action = None
                keystroke = self.stdscr.getch()
                if keystroke != curses.ERR:
                    if keystroke in (KEY_S, KEY_W, KEY_UP_ARROW, KEY_DOWN_ARROW):
                        action = KEYSTROKE_TO_ACTION[keystroke]

                self.game.tick(action)
                self._render()

            self._game_over()
        except Exception:
            curses.endwin()
            self.logger.output_log_messages()
            raise

    @staticmethod
    def _init_grid() -> Grid:
        """Initialises the grid with zeros"""

        grid = np.zeros((GRID_SIZE_Y, GRID_SIZE_X), dtype=np.float32)
        grid[:, [0]] = DEATH_ZONE_LEFT  # create death zone
        grid[:, [-1]] = DEATH_ZONE_RIGHT  # create death zone
        grid[[0], :] = CEILING
        grid[[-1], :] = FLOOR

        return grid


def main(stdscr: curses.window) -> None:
    """Main function"""

    logger = GameLogger()
    game = Game(logger, single_player=True)
    terminal_app = TerminalApp(stdscr, game, logger)
    terminal_app.run()


if __name__ == "__main__":
    curses.wrapper(main)
