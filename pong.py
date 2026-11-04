"""A CLI Pong game"""
import os
import random
import curses
import sys
import time
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import numpy.typing as npt

Grid = npt.NDArray[np.float32]

RANDOM = np.random.default_rng()

ITERATIONS = 1000
TICK_SLEEP_TIME = 1 / 30

GRID_SIZE_X = 30
GRID_SIZE_Y = 10

LEFT_PADDLE_X_VALUE = 1
RIGHT_PADDLE_X_VALUE = GRID_SIZE_X - 2

EMPTY_SPACE = 0
DEATH_ZONE_LEFT = 1
DEATH_ZONE_RIGHT = 2
FLOOR = 3
CEILING = 4
LEFT_PADDLE = 5
RIGHT_PADDLE = 6
BALL = 9

BOUNCING_OBJECTS = [FLOOR, CEILING, LEFT_PADDLE, RIGHT_PADDLE]
DEATH_ZONES = [DEATH_ZONE_LEFT, DEATH_ZONE_RIGHT]

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

# Maps object to unit normal vector, used in the calculation of velocity after a bounce.
OBJECT_TO_UNIT_NORMAL_VECTOR = {
    DEATH_ZONE_LEFT: np.array((0, 1)),  # Vertical wall (left)
    DEATH_ZONE_RIGHT: np.array((0, -1)),  # Vertical wall (right)
    FLOOR: np.array((1, 0)),  # Horizontal floor
    CEILING: np.array((-1, 0)),  # Horizontal ceiling
    LEFT_PADDLE: np.array((0, 1)),
    RIGHT_PADDLE: np.array((0, -1)),
}

KEY_S = 119
KEY_W = 115
KEY_UP_ARROW = 259
KEY_DOWN_ARROW = 258

STARTING_LIVES = 3

MIN_STARTING_Y_VELOCITY = 1
MAX_STARTING_Y_VELOCITY = 10
MIN_STARTING_X_VELOCITY = 20
MAX_STARTING_X_VELOCITY = 40


class GameOver(Exception):
    """Raised when a player loses all their lives to end the game."""


@dataclass
class State:
    position_y: float
    position_x: float
    velocity_y: float
    velocity_x: float

    left_paddle_position_y: float
    right_paddle_position_y: float
    left_paddle_velocity: float
    right_paddle_velocity: float

    grid: Grid

    left_lives: int
    right_lives: int

    @property
    def rounded_ball_position(self) -> Tuple[int, int]:
        """Rounds the ball position to the nearest integer for render"""

        return round(self.position_y), round(self.position_x)

    @property
    def left_paddle_position(self) -> Tuple[int, int]:
        return round(self.left_paddle_position_y), LEFT_PADDLE_X_VALUE

    @property
    def right_paddle_position(self) -> Tuple[int, int]:
        return round(self.right_paddle_position_y), RIGHT_PADDLE_X_VALUE


def render(stdscr: curses.window, state: State) -> None:
    """Prints the provided grid, translating numbers into icons."""

    grid = state.grid

    frame = "\n".join("".join(DISPLAY_CHARACTERS[cell] for cell in row) for row in grid)
    stdscr.addstr(0, 0, frame)

    score = f"Left {state.left_lives} : {state.right_lives} Right".center(GRID_SIZE_X)
    stdscr.addstr(len(grid), 0, score)
    # stdscr.refresh()


def init_grid() -> Grid:
    """Initialises the grid with zeros"""

    grid = np.zeros((GRID_SIZE_Y, GRID_SIZE_X), dtype=np.float32)
    grid[:, [0]] = DEATH_ZONE_LEFT  # create death zone
    grid[:, [-1]] = DEATH_ZONE_RIGHT  # create death zone
    grid[[0], :] = CEILING
    grid[[-1], :] = FLOOR

    return grid


def ball_to_start_position(state: State) -> None:
    """Sets the ball in the middle of the frame and randomises the starting velocity"""

    # y velocity in positions per tick - positive means moving top to bottom
    state.velocity_y = (random.randint(MIN_STARTING_Y_VELOCITY, MAX_STARTING_Y_VELOCITY) * random.choice([1, -1])) / 100
    # x velocity in positions per tick - positive means moving left to right
    state.velocity_x = (random.randint(MIN_STARTING_X_VELOCITY, MAX_STARTING_X_VELOCITY) * random.choice([1, -1])) / 100

    state.position_y = state.grid.shape[0] // 2
    state.position_x = state.grid.shape[1] // 2


def init_state() -> State:
    """Initialises inital state"""

    grid = init_grid()

    paddle_y_pos = grid.shape[0] // 2

    state = State(
        position_y=0,  # will be set before the game starts in ball_to_start_position, same below
        position_x=0,
        velocity_y=0,
        velocity_x=0,
        left_paddle_position_y=paddle_y_pos,
        right_paddle_position_y=paddle_y_pos,
        left_paddle_velocity=0,
        right_paddle_velocity=0,
        grid=grid,
        left_lives=STARTING_LIVES,
        right_lives=STARTING_LIVES
    )

    ball_to_start_position(state)
    return state


def bounce(state: State, bouncing_object: int, log_messages: List[str], tick_index: int) -> None:
    """
    Bounces the ball, calculates the resulting velocity of the ball after the bounce.
    Calculates resulting velocity using the equation:

            vout = vin - 2(vin . N)N

    Where N is the unit normal vector of the surface.
    """

    # Calculate new velocity
    unit_normal_vector = OBJECT_TO_UNIT_NORMAL_VECTOR[bouncing_object]
    input_velocity = np.array([state.velocity_y, state.velocity_x], dtype=np.float32)
    state.velocity_y, state.velocity_x = input_velocity - (2 * np.dot(input_velocity, unit_normal_vector) * unit_normal_vector)

    log_messages.append(f"{tick_index}: bouncing, input velocity: {list(input_velocity)}, output velocity: {[state.velocity_y, state.velocity_x]}")

    # Send the ball off in the new velocity
    state.position_y = state.position_y + state.velocity_y
    state.position_x = state.position_x + state.velocity_x


def register_life_lost(state: State, bouncing_object: int, log_messages: List[str], tick_index: int) -> None:
    """Registers a life lost when hitting the death zone."""

    if bouncing_object == DEATH_ZONE_LEFT:
        state.left_lives = state.left_lives - 1
        log_messages.append(f"{tick_index}: left life lost, remaining lives: {state.left_lives}")
    elif bouncing_object == DEATH_ZONE_RIGHT:
        state.right_lives = state.right_lives - 1
        log_messages.append(f"{tick_index}: right life lost, remaining lives: {state.right_lives}")
    else:
        raise ValueError(f"Unhandled bouncing object: {bouncing_object}")

    ball_to_start_position(state)


def register_keystroke(state: State, keystroke: int) -> None:
    """Registers keystroke in state"""

    if keystroke == KEY_W:
        state.left_paddle_position_y = max(min(state.left_paddle_position_y + 1, state.grid.shape[0] - 2), 1)
    elif keystroke == KEY_S:
        state.left_paddle_position_y = max(min(state.left_paddle_position_y - 1, state.grid.shape[0] - 2), 1)
    elif keystroke == KEY_UP_ARROW:
        state.right_paddle_position_y = max(min(state.right_paddle_position_y - 1, state.grid.shape[0] - 2), 1)
    elif keystroke == KEY_DOWN_ARROW:
        state.right_paddle_position_y = max(min(state.right_paddle_position_y + 1, state.grid.shape[0] - 2), 1)
    else:
        raise Exception(f"Unahndled keystroke: {keystroke}")


def tick(state: State, log_messages: List[str], tick_index: int) -> None:
    """Updates the state with a single new tick, mutates state in place"""

    log_messages.append(f"{tick_index}: starting tick")

    # --- Update the state ---

    # Calculate the new ball positions
    potential_new_y, potential_new_x = state.position_y + state.velocity_y, state.position_x + state.velocity_x
    rounded_potential_new_ball_position = round(potential_new_y), round(potential_new_x)
    log_messages.append(f"{tick_index}: rounded potential new ball position: {rounded_potential_new_ball_position}")

    next_position_object = state.grid[rounded_potential_new_ball_position]
    if next_position_object in BOUNCING_OBJECTS:
        log_messages.append(f"{tick_index}: bounce detected, bounce object is {next_position_object}")
        bounce(state, next_position_object, log_messages, tick_index)
    elif next_position_object in DEATH_ZONES:
        log_messages.append(f"{tick_index}: death zone detected detected, death zone object is {next_position_object}")
        register_life_lost(state, next_position_object, log_messages, tick_index)
    else:
        state.position_y, state.position_x = potential_new_y, potential_new_x

    log_messages.append(
        f"{tick_index}: resulting position is {state.rounded_ball_position}, new left paddle position {state.left_paddle_position_y}, new right paddle position "
        f"{state.right_paddle_position_y}"
    )

    # --- Update the grid ---

    # wipe the previous positions of ball and paddles
    state.grid[np.isin(state.grid, [LEFT_PADDLE, RIGHT_PADDLE, BALL])] = EMPTY_SPACE

    # render the new ones
    state.grid[state.rounded_ball_position] = BALL
    state.grid[state.left_paddle_position] = LEFT_PADDLE
    state.grid[state.right_paddle_position] = RIGHT_PADDLE


def splash_winning_screen(stdscr: curses.window, state: State) -> None:
    """Displays the winner and the final score."""

    winner, loser = ("Left", "Right") if state.right_lives == 0 else ("Right", "Left")
    lines = [
        "GAME OVER",
        "",
        f"{winner} wins.",
        f"{loser} loses.",
        "",
        f"Left {state.left_lives} : {state.right_lives} Right",
        "",
        "Press any key",
    ]

    window = curses.newwin(GRID_SIZE_Y, GRID_SIZE_X, 0, 0)
    window.box()
    for row, line in enumerate(lines):
        window.addstr(row + 1, (GRID_SIZE_X - len(line)) // 2, line)

    stdscr.clear()
    stdscr.refresh()
    window.refresh()

    stdscr.nodelay(False)
    stdscr.getch()


def check_game_over(stdscr: curses.window, state: State) -> None:
    """Checks if the game is over, displays message to user if so."""

    if state.left_lives == 0 or state.right_lives == 0:
        # Someones won
        splash_winning_screen(stdscr, state)
        raise GameOver


def run_main_loop(stdscr: curses.window, state: State, log_messages: List[str]) -> None:
    """Executes the main loop until completion."""

    stdscr.nodelay(True)

    for i in range(ITERATIONS):
        time.sleep(TICK_SLEEP_TIME)

        keystroke = stdscr.getch()
        if keystroke != curses.ERR:
            log_messages.append(f"Keystroke found {keystroke}")
            if keystroke in (KEY_W, KEY_S, KEY_UP_ARROW, KEY_DOWN_ARROW):
                register_keystroke(state, keystroke)

        tick(state, log_messages, i)
        check_game_over(stdscr, state)
        render(stdscr, state)


def main() -> None:
    """Entry point"""

    state = init_state()
    log_messages = []

    try:
        curses.wrapper(lambda stdscr: run_main_loop(stdscr, state, log_messages))
    except GameOver:
        for log_message in log_messages:
            print(log_message)
        print("Game finished!")
    except Exception as error:
        for log_message in log_messages:
            print(log_message)

        raise error
    except KeyboardInterrupt:
        print("Exiting cleanly")


if __name__ == "__main__":
    main()
