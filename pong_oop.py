import curses
import sys
import time
import random
from enum import IntEnum
from typing import List, Tuple, Optional

import numpy as np

from pong_procedural import TICK_SLEEP_TIME, KEY_S, KEY_W, KEY_UP_ARROW, KEY_DOWN_ARROW, GRID_SIZE_Y, GRID_SIZE_X, DEATH_ZONE_LEFT, DEATH_ZONE_RIGHT, \
    CEILING, FLOOR, DISPLAY_CHARACTERS, STARTING_LIVES, LEFT_PADDLE_X_VALUE, RIGHT_PADDLE_X_VALUE, LEFT_PADDLE, RIGHT_PADDLE, BALL, BOUNCING_OBJECTS, \
    OBJECT_TO_UNIT_NORMAL_VECTOR, init_grid

STARTING_Y_VELOCITY = 0.1
STARTING_X_VELOCITY = 0.3

BALL_X_ACCELERATION = 0.05  # ball accelerates this amount per bounce on a paddle
BALL_Y_ACCELERATION = 0.0125

class PaddleSide(IntEnum):
    LEFT = 0
    RIGHT = 1


class Paddle:
    """A paddle"""

    def __init__(self, side: PaddleSide, starting_y: int) -> None:
        self.side = side
        self.position_y = starting_y

    @property
    def position_x(self) -> int:
        """Returns the x position value based on side"""
        return RIGHT_PADDLE_X_VALUE if self._get_side_safe() == PaddleSide.RIGHT else LEFT_PADDLE_X_VALUE

    @property
    def position(self) -> Tuple[int, int]:
        """Returns the paddle position in format y, x"""
        return self.position_y, self.position_x

    @property
    def display_character(self) -> str:
        """Returns the display character to render"""
        return DISPLAY_CHARACTERS[RIGHT_PADDLE] if self._get_side_safe() == PaddleSide.RIGHT else DISPLAY_CHARACTERS[LEFT_PADDLE]

    def _get_side_safe(self) -> PaddleSide:
        """Returns the side after validating"""

        if self.side not in [PaddleSide.LEFT, PaddleSide.RIGHT]:
            raise ValueError(f"Invalid side: {self.side}")

        return self.side

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

    def bounce(self, unit_normal_vector: Tuple[int, int]) -> None:
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

    def __init__(self, stdscr: curses.window) -> None:
        self.ball = Ball()
        paddle_starting_y = GRID_SIZE_Y // 2
        self.left_paddle = Paddle(PaddleSide.LEFT, starting_y=paddle_starting_y)
        self.right_paddle = Paddle(PaddleSide.RIGHT, starting_y=paddle_starting_y)

        self.left_lives = STARTING_LIVES
        self.right_lives = STARTING_LIVES

        self.stdscr = stdscr
        self.stdscr.nodelay(True)
        self.tick_index = 0
        self.log_messages: List[str] = []

        self.grid = init_grid()

    def _log(self, message: str) -> None:
        """Appends log message"""
        self.log_messages.append(f"{self.tick_index}: {message}")

    def _render(self) -> None:
        """Renders the grid"""

        # overlay grid
        frame = "\n".join("".join(DISPLAY_CHARACTERS[cell] for cell in row) for row in self.grid)
        self.stdscr.addstr(0, 0, frame)

        # overlay paddles
        self.stdscr.addstr(self.left_paddle.position[0], self.left_paddle.position[1], self.left_paddle.display_character)
        self.stdscr.addstr(self.right_paddle.position[0], self.right_paddle.position[1], self.right_paddle.display_character)

        # overlay ball
        self.stdscr.addstr(self.ball.rounded_position[0], self.ball.rounded_position[1], self.ball.display_character)

        # overlay score
        score = f"Left {self.left_lives} : {self.right_lives} Right".center(GRID_SIZE_X)
        self.stdscr.addstr(len(self.grid), 0, score)
        self.stdscr.refresh()

    def _register_keystroke(self, keystroke: int) -> None:
        """Responds to a keystroke by updating the paddle state"""

        if keystroke == KEY_W:
            self.left_paddle.move_paddle(-1)
        elif keystroke == KEY_S:
            self.left_paddle.move_paddle(1)
        elif keystroke == KEY_UP_ARROW:
            self.right_paddle.move_paddle(-1)
        elif keystroke == KEY_DOWN_ARROW:
            self.right_paddle.move_paddle(1)
        else:
            raise ValueError(f"Unhandled keystroke: {keystroke}")

    @staticmethod
    def _collides_with_paddle(ball_position_y: int, ball_position_x: int, paddle_position_y: int, paddle_position_x: int) -> bool:
        """Checks if the ball collides with the paddle"""
        return ball_position_x == paddle_position_x and ball_position_y == paddle_position_y

    def _check_collision(self) -> Optional[int]:
        """Checks if the next ball position will collide with an object"""

        next_position_y, next_position_x = self.ball.next_position

        if next_position_y == 0:
            return CEILING
        elif next_position_y == GRID_SIZE_Y - 1:
            return FLOOR
        elif next_position_x == 0:
            return DEATH_ZONE_LEFT
        elif next_position_x == GRID_SIZE_X - 1:
            return DEATH_ZONE_RIGHT
        elif self._collides_with_paddle(next_position_y, next_position_x, self.left_paddle.position_y, self.left_paddle.position_x):
            return LEFT_PADDLE
        elif self._collides_with_paddle(next_position_y, next_position_x, self.right_paddle.position_y, self.right_paddle.position_x):
            return RIGHT_PADDLE
        else:
            return None

    def _game_over(self) -> None:
        """Splashes a screen to say game over."""
        winner, loser = ("Left", "Right") if self.right_lives == 0 else ("Right", "Left")
        lines = [
            "GAME OVER",
            "",
            f"{winner} wins.",
            f"{loser} loses.",
            "",
            f"Left {self.left_lives} : {self.right_lives} Right",
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

    def _handle_life_lost(self, death_zone: int) -> None:
        """Handles a player losing a life"""

        if death_zone == DEATH_ZONE_LEFT:
            self.left_lives -= 1
        elif death_zone == DEATH_ZONE_RIGHT:
            self.right_lives -= 1
        else:
            raise ValueError(f"Invalid object for losing a life: {death_zone}")

        if self.left_lives == 0 or self.right_lives == 0:
            self._game_over()

        self.ball.reset_to_start()

    def _tick(self) -> None:
        """Ticks the game's state by one increment"""

        self._log("Starting tick")

        # Update ball position
        collision_value = self._check_collision()
        if collision_value in BOUNCING_OBJECTS:
            self.ball.bounce(OBJECT_TO_UNIT_NORMAL_VECTOR[collision_value])
            if collision_value == LEFT_PADDLE:
                self.ball.accelerate(BALL_Y_ACCELERATION, BALL_X_ACCELERATION)
            elif collision_value == RIGHT_PADDLE:
                self.ball.accelerate(-BALL_Y_ACCELERATION, -BALL_X_ACCELERATION)
        elif collision_value in [DEATH_ZONE_LEFT, DEATH_ZONE_RIGHT]:
            # self.ball.bounce(OBJECT_TO_UNIT_NORMAL_VECTOR[collision_value])
            self._handle_life_lost(collision_value)
        else:
            self.ball.tick()

    def _main_loop(self) -> None:
        """Runs the main loop to game completion"""

        while True:
            time.sleep(TICK_SLEEP_TIME)

            keystroke = self.stdscr.getch()
            if keystroke != curses.ERR:
                self._log(f"Keystroke found {keystroke}")
                if keystroke in (KEY_S, KEY_W, KEY_UP_ARROW, KEY_DOWN_ARROW):
                    self._register_keystroke(keystroke)

            self._tick()
            self._render()
            self.tick_index += 1

    def run(self) -> None:
        """Runs the game, starts the main loop."""

        try:
            self._main_loop()
        except Exception:
            curses.endwin()
            for log_message in self.log_messages:
                print(log_message)

def main(stdscr: curses.window) -> None:
    """Main function"""

    game = Game(stdscr)
    game.run()

if __name__ == "__main__":
    curses.wrapper(main)
