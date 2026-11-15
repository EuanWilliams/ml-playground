"""Interface for playing pong via the terminal"""

import argparse
import curses
import time
from enum import IntEnum
from typing import Dict, Optional

import numpy as np
import numpy.typing as npt

from pong.controllers import KeyboardController, Controller, AutoController, Observation, ModelController
from pong.game.game import Game, GameLogger, PlayerAction, Player, PaddleMove
from pong.game.game_constants import EMPTY_SPACE, DEATH_ZONE_LEFT, DEATH_ZONE_RIGHT, FLOOR, CEILING, LEFT_PADDLE, RIGHT_PADDLE, BALL, GRID_SIZE_Y, GRID_SIZE_X
from pong.training.model_common import get_observation

Grid = npt.NDArray[np.float32]

# TUI Game
MODE_MANUAL = "manual"
MODE_AUTO = "auto"
MODE_MODEL = "model"

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


class GameResult(IntEnum):
    """Enum for the result of the game"""

    NO_RESULT = 0
    COMPLETE = 1
    USER_STOPPED = 2
    ERROR = 3


class TerminalApp:
    """Terminal app for running pong interactively in the terminal"""

    def __init__(self, stdscr: curses.window, game: Game, controllers: Dict[Player, Controller], logger: GameLogger) -> None:
        self.stdscr = stdscr
        self.game = game
        self.logger = logger
        self.single_player = self.game.single_player

        self.controllers = controllers

        if self.single_player and len(self.controllers) > 1:
            raise ValueError("Multiple controllers provided in single player mode")
        if not self.single_player and len(self.controllers) != 2:
            raise ValueError("Single controller provided in multiplayer mode")

        self.grid = self._init_grid()

    def _game_over(self) -> None:
        """Splashes a screen to say game over."""
        if self.single_player:
            lines = [
                "GAME OVER",
                "",
                f"Final score: {self.game.score}",
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
        frame = "\n".join("".join(DISPLAY_CHARACTERS[cell] for cell in row if cell != BALL) for row in self.grid)
        self.stdscr.addstr(0, 0, frame)

        # overlay paddles
        self.stdscr.addstr(self.game.left_paddle.position[0], self.game.left_paddle.position[1], DISPLAY_CHARACTERS[LEFT_PADDLE])
        if not self.single_player:
            assert self.game.right_paddle is not None
            self.stdscr.addstr(self.game.right_paddle.position[0], self.game.right_paddle.position[1], DISPLAY_CHARACTERS[RIGHT_PADDLE])

        # overlay ball
        for ball in self.game.balls:
            self.stdscr.addstr(ball.rounded_position[0], ball.rounded_position[1], DISPLAY_CHARACTERS[BALL])

        # overlay bottom line
        if self.game.last_life_lost_tick is not None and self.game.tick_index - self.game.last_life_lost_tick <= 25:
            bottom_line = "Life lost (×_×)".center(GRID_SIZE_X)
        elif self.single_player:
            bottom_line = f"Current score: {self.game.score} | Lives: {self.game.left_lives}".center(GRID_SIZE_X)
        else:
            bottom_line = f"Left {self.game.left_lives} : {self.game.right_lives} Right".center(GRID_SIZE_X)

        self.stdscr.addstr(len(self.grid), 0, bottom_line)
        self.stdscr.refresh()

    def _keystroke_to_action(self, keystroke: int) -> PlayerAction:
        """Converts the keystroke to an action"""

        if keystroke == KEY_W:
            return PlayerAction.LEFT_PADDLE_UP
        elif keystroke == KEY_S:
            return PlayerAction.LEFT_PADDLE_DOWN
        elif keystroke == KEY_UP_ARROW:
            if self.single_player:
                return PlayerAction.LEFT_PADDLE_UP
            else:
                return PlayerAction.RIGHT_PADDLE_UP
        elif keystroke == KEY_DOWN_ARROW:
            if self.single_player:
                return PlayerAction.LEFT_PADDLE_DOWN
            else:
                return PlayerAction.RIGHT_PADDLE_DOWN
        else:
            raise ValueError(f"Unhandled keystroke: {keystroke}")

    def run(self) -> None:
        """Runs the game, render status to terminal"""

        game_result = GameResult.ERROR  # assume exception
        game_error: Optional[BaseException] = None
        try:
            self.stdscr.nodelay(True)

            while self.game.winner is None:
                time.sleep(TICK_SLEEP_TIME)

                actions = self._retrieve_controller_actions()

                self.game.tick(actions)
                self._render()

            self._game_over()
            game_result = GameResult.COMPLETE
        except KeyboardInterrupt:
            game_result = GameResult.USER_STOPPED
        except Exception as error:
            game_result = GameResult.ERROR
            game_error = error
        finally:
            curses.endwin()
            if game_result == GameResult.ERROR:
                self.logger.output_log_messages()
                assert game_error is not None
                raise game_error
            elif game_result == GameResult.COMPLETE:
                if self.single_player:
                    print(f"Final score: {self.game.score}")
                else:
                    print(f"Final scores: Left {self.game.left_lives} : {self.game.right_lives} Right")
            else:
                print("Game stopped by user.")

    def _retrieve_controller_actions(self) -> Dict[Player, PaddleMove]:
        """Retrieves the controller's actions"""

        keystroke = self.stdscr.getch()

        actions = {}
        for player, controller in self.controllers.items():
            controller.read_observation(self.game, extra_data={"keystroke": keystroke})
            action = controller.get_move()
            actions[player] = action

        return actions

    @staticmethod
    def _init_grid() -> Grid:
        """Initialises the grid with zeros"""

        grid = np.zeros((GRID_SIZE_Y, GRID_SIZE_X), dtype=np.float32)
        grid[:, [0]] = DEATH_ZONE_LEFT  # create death zone
        grid[:, [-1]] = DEATH_ZONE_RIGHT  # create death zone
        grid[[0], :] = CEILING
        grid[[-1], :] = FLOOR

        return grid


def main(stdscr: curses.window, opt: argparse.Namespace) -> None:
    """Main function"""

    logger = GameLogger()

    if opt.multiplayer:
        raise ValueError("Multiplayer mode is not yet implemented in the terminal app.")

    if opt.mode == MODE_AUTO:
        controllers: Dict[Player, Controller] = {Player.LEFT: AutoController(logger)}
    elif opt.mode == MODE_MODEL:
        controllers = {Player.LEFT: ModelController(logger)}
    else:
        controllers = {Player.LEFT: KeyboardController(key_press_up=[KEY_W, KEY_UP_ARROW], key_press_down=[KEY_S, KEY_DOWN_ARROW])}

    random_generator = np.random.default_rng()
    game = Game(random=random_generator, single_player=True, logger=logger)
    terminal_app = TerminalApp(stdscr, game, controllers, logger)
    terminal_app.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("TUI for pong")
    parser.add_argument("--mode", type=str, default=MODE_AUTO, choices=[MODE_AUTO, MODE_MANUAL, MODE_MODEL], help="Which mode to play in.")
    parser.add_argument("--multiplayer", action="store_true", help="Play in multiplayer mode.")
    args = parser.parse_args()

    curses.wrapper(main, args)
