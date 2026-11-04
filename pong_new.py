import curses
import time
from typing import List, Tuple

import numpy as np

from pong import TICK_SLEEP_TIME, KEY_S, KEY_W, KEY_UP_ARROW, KEY_DOWN_ARROW, GRID_SIZE_Y, Grid, GRID_SIZE_X, DEATH_ZONE_LEFT, DEATH_ZONE_RIGHT, CEILING, FLOOR, \
    init_grid, DISPLAY_CHARACTERS, STARTING_LIVES


class Paddle:
    """A paddle"""

    def __init__(self) -> None:
        self.position_y = 0

    def move_paddle(self, increment: int) -> None:
        """Moves the ball a single increment"""
        self.position_y = max(min(self.position_y + increment, GRID_SIZE_Y - 2), 1)


class Ball:
    """The ball"""

    def __init__(self) -> None:
        self.position_x = 0.0
        self.position_y = 0.0

        self.velocity_y = 0.0
        self.velocity_x = 0.0

    def set_position(self, position: Tuple[int, int]) -> None:
        """Sets the balls position"""

        self.position_y, self.position_x = position


class Game:
    """The game"""

    def __init__(self, stdscr: curses.window) -> None:
        self.ball = Ball()
        self.left_paddle = Paddle()
        self.right_paddle = Paddle()

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

    def _register_keystroke(self, keystroke: int) -> None:
        """Responds to a keystroke by updating the paddle state"""

        if keystroke == KEY_W:
            self.left_paddle.move_paddle(1)
        elif keystroke == KEY_S:
            self.left_paddle.move_paddle(-1)
        elif keystroke == KEY_UP_ARROW:
            self.right_paddle.move_paddle(1)
        elif keystroke == KEY_DOWN_ARROW:
            self.right_paddle.move_paddle(-1)
        else:
            raise ValueError(f"Unhandled keystroke: {keystroke}")

    def _tick(self) -> None:
        """Ticks the game's state by one increment"""

        self._log("Starting tick")

    def _render(self) -> None:
        """Renders the grid"""

        frame = "\n".join("".join(DISPLAY_CHARACTERS[cell] for cell in row) for row in self.grid)
        self.stdscr.addstr(0, 0, frame)

        score = f"Left {self.left_lives} : {self.right_lives} Right".center(GRID_SIZE_X)
        self.stdscr.addstr(len(self.grid), 0, score)
        self.stdscr.refresh()

    def run(self) -> None:
        """Runs the game, starts the main loop."""

        while True:
            time.sleep(TICK_SLEEP_TIME)

            # keystroke = self.stdscr.getch()
            # if keystroke != curses.ERR:
            #     self._log(f"Keystroke found {keystroke}")
            #     if keystroke in (KEY_S, KEY_W, KEY_UP_ARROW, KEY_DOWN_ARROW):
            #         self._register_keystroke(keystroke)

            self._tick()
            self.tick_index += 1
            self._render()


def main(stdscr: curses.window) -> None:
    """Main function"""

    game = Game(stdscr)
    game.run()

if __name__ == "__main__":
    curses.wrapper(main)
