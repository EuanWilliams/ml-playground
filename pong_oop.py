import curses
import sys
import time
import random
from enum import IntEnum
from typing import List, Tuple, Optional

import numpy as np
import numpy.typing as npt

from pong_procedural import (
    init_grid,
)
from pong_constants import TICK_SLEEP_TIME, GRID_SIZE_X, GRID_SIZE_Y, LEFT_PADDLE_X_VALUE, RIGHT_PADDLE_X_VALUE, DEATH_ZONE_LEFT, DEATH_ZONE_RIGHT, FLOOR, \
    CEILING, LEFT_PADDLE, RIGHT_PADDLE, BALL, BOUNCING_OBJECTS, DISPLAY_CHARACTERS, OBJECT_TO_UNIT_NORMAL_VECTOR, KEY_W, KEY_S, KEY_UP_ARROW, KEY_DOWN_ARROW, \
    STARTING_LIVES

STARTING_Y_VELOCITY = 0.1
STARTING_X_VELOCITY = 0.3

BALL_X_ACCELERATION = 0.025  # ball accelerates this amount per bounce on a paddle
BALL_Y_ACCELERATION = 0.0625


class Player(IntEnum):
    LEFT = 0
    RIGHT = 1


class PlayerAction(IntEnum):
    LEFT_PADDLE_UP = 0
    LEFT_PADDLE_DOWN = 1
    RIGHT_PADDLE_UP = 2
    RIGHT_PADDLE_DOWN = 3


KEYSTROKE_TO_ACTION = {
    KEY_W: PlayerAction.LEFT_PADDLE_UP,
    KEY_S: PlayerAction.LEFT_PADDLE_DOWN,
    KEY_UP_ARROW: PlayerAction.RIGHT_PADDLE_UP,
    KEY_DOWN_ARROW: PlayerAction.RIGHT_PADDLE_DOWN,
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

        self.velocity_y += delta_y
        self.velocity_x += delta_x

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

    def __init__(self, logger: GameLogger) -> None:
        self.logger = logger

        self.ball = Ball()
        paddle_starting_y = GRID_SIZE_Y // 2
        self.left_paddle = Paddle(Player.LEFT, starting_y=paddle_starting_y)
        self.right_paddle = Paddle(Player.RIGHT, starting_y=paddle_starting_y)

        self.left_lives = STARTING_LIVES
        self.right_lives = STARTING_LIVES
        self.winner: Optional[Player] = None

        self.tick_index = 0
        self.log_messages: List[str] = []

        self.grid = init_grid()

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
        elif player_action == PlayerAction.RIGHT_PADDLE_DOWN:
            if self.right_paddle is not None:
                self.right_paddle.move_paddle(1)
        else:
            raise ValueError(f"Unhandled player action: {player_action}")

    @staticmethod
    def _collides_with_paddle(ball_position_y: int, ball_position_x: int, paddle_position_y: int, paddle_position_x: int) -> bool:
        """Checks if the ball collides with the paddle"""
        return ball_position_x == paddle_position_x and ball_position_y == paddle_position_y

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
        elif self._collides_with_paddle(next_position_y, next_position_x, self.left_paddle.position_y, self.left_paddle.position_x):
            return LEFT_PADDLE
        elif self._collides_with_paddle(next_position_y, next_position_x, self.right_paddle.position_y, self.right_paddle.position_x):
            return RIGHT_PADDLE
        else:
            return None

    def _handle_life_lost(self, death_zone: int) -> None:
        """Handles a player losing a life"""

        self._log(f"Life lost to death zone {death_zone}")

        if death_zone == DEATH_ZONE_LEFT:
            self.left_lives -= 1
        elif death_zone == DEATH_ZONE_RIGHT:
            self.right_lives -= 1
        else:
            raise ValueError(f"Invalid object for losing a life: {death_zone}")

        if self.left_lives == 0:
            self.winner = Player.LEFT
        elif self.right_lives == 0:
            self.winner = Player.RIGHT
        else:
            self.ball.reset_to_start()

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
            if collision_value == LEFT_PADDLE:
                self.ball.accelerate(BALL_Y_ACCELERATION, BALL_X_ACCELERATION)
            elif collision_value == RIGHT_PADDLE:
                self.ball.accelerate(-BALL_Y_ACCELERATION, -BALL_X_ACCELERATION)
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

    def _game_over(self) -> None:
        """Splashes a screen to say game over."""
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

        sys.exit()

    def _render(self) -> None:
        """Renders the current status to terminal"""

        # overlay grid
        frame = "\n".join("".join(DISPLAY_CHARACTERS[cell] for cell in row) for row in self.game.grid)
        self.stdscr.addstr(0, 0, frame)

        # overlay paddles
        self.stdscr.addstr(self.game.left_paddle.position[0], self.game.left_paddle.position[1], self.game.left_paddle.display_character)
        self.stdscr.addstr(self.game.right_paddle.position[0], self.game.right_paddle.position[1], self.game.right_paddle.display_character)

        # overlay ball
        self.stdscr.addstr(self.game.ball.rounded_position[0], self.game.ball.rounded_position[1], self.game.ball.display_character)

        # overlay score
        score = f"Left {self.game.left_lives} : {self.game.right_lives} Right".center(GRID_SIZE_X)
        self.stdscr.addstr(len(self.game.grid), 0, score)
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

            curses.endwin()
            print(f"Final score: Left {self.game.left_lives} : {self.game.right_lives} Right")
        except Exception:
            curses.endwin()
            self.logger.output_log_messages()
            raise


def main(stdscr: curses.window) -> None:
    """Main function"""

    logger = GameLogger()
    game = Game(logger)
    terminal_app = TerminalApp(stdscr, game, logger)
    terminal_app.run()


if __name__ == "__main__":
    curses.wrapper(main)
