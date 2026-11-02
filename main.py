import random
import sys
import time
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import numpy.typing as npt

Grid = npt.NDArray[np.uint8]

RANDOM = np.random.default_rng()

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

BOUNCING_OBJECTS = [DEATH_ZONE_LEFT, DEATH_ZONE_RIGHT, FLOOR, CEILING]

DISPLAY_CHARACTERS = {
    EMPTY_SPACE: ' ',
    DEATH_ZONE_LEFT: "|",
    DEATH_ZONE_RIGHT: "|",
    CEILING: "-",
    FLOOR: "-",
    BALL: "0",
    LEFT_PADDLE: "[",
    RIGHT_PADDLE: "]",
}

HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"
CLEAR_SCREEN = "\033[2J"
CURSOR_HOME = "\033[H"

# Maps object to unit normal vector, used in the calculation of velocity after a bounce.
OBJECT_TO_UNIT_NORMAL_VECTOR = {
    DEATH_ZONE_LEFT: np.array((0, 1)),  # Vertical wall (left)
    DEATH_ZONE_RIGHT: np.array((0, -1)),  # Vertical wall (right)
    FLOOR: np.array((1, 0)),  # Horizontal floor
    CEILING: np.array((-1, 0))  # Horizontal ceiling
}


@dataclass
class State:
    position_y: float
    position_x: float
    velocity_y: float
    velocity_x: float

    left_paddle_position_y: int
    right_paddle_position_y: int

    grid: Grid

    @property
    def rounded_ball_position(self) -> Tuple[int, int]:
        """Rounds the ball position to the nearest integer for render"""

        return round(self.position_y), round(self.position_x)

    @property
    def left_paddle_position(self) -> Tuple[int, int]:
        return self.left_paddle_position_y, LEFT_PADDLE_X_VALUE

    @property
    def right_paddle_position(self) -> Tuple[int, int]:
        return self.right_paddle_position_y, RIGHT_PADDLE_X_VALUE


def render(state: State) -> None:
    """Prints the provided grid, translating numbers into icons."""

    grid = state.grid

    frame = "\n".join("".join(DISPLAY_CHARACTERS[cell] for cell in row) for row in grid)
    sys.stdout.write(CURSOR_HOME + frame)
    sys.stdout.flush()

def init_grid() -> Grid:
    """Initialises the grid with zeros"""

    grid = np.zeros((GRID_SIZE_Y, GRID_SIZE_X), dtype=np.float32)
    grid[:, [0]] = DEATH_ZONE_LEFT  # create death zone
    grid[:, [-1]] = DEATH_ZONE_RIGHT  # create death zone
    grid[[0], :] = CEILING
    grid[[-1], :] = FLOOR

    return grid

def init_state() -> State:
    """Initialises inital state"""

    grid = init_grid()

    # y velocity in positions per tick - positive means moving top to bottom
    initial_ball_velocity_y = random.randint(0, 100) / 100
    # x velocity in positions per tick - positive means moving left to right
    initial_ball_velocity_x = random.randint(0, 100) / 100

    paddle_y_pos = grid.shape[0] // 2

    return State(
        position_y=grid.shape[0] // 2,
        position_x=grid.shape[1] // 2,
        velocity_y=initial_ball_velocity_y,
        velocity_x=initial_ball_velocity_x,
        left_paddle_position_y=paddle_y_pos,
        right_paddle_position_y=paddle_y_pos,
        grid=grid
    )

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

def tick(state: State, log_messages: List[str], tick_index: int) -> None:
    """Updates the state with a single new tick, mutates state in place"""

    log_messages.append(f"{tick_index}: starting tick")

    # --- Update the state ---

    # Calculate the new ball positions
    potential_new_y, potential_new_x = state.position_y + state.velocity_y, state.position_x + state.velocity_x
    rounded_potential_new_ball_position = round(potential_new_y), round(potential_new_x)
    log_messages.append(f"{tick_index}: rounded potential new ball position: {rounded_potential_new_ball_position}")

    # Check if we should bounce!
    if state.grid[rounded_potential_new_ball_position] in BOUNCING_OBJECTS:
        log_messages.append(f"{tick_index}: bounce detected, bounce object is {state.grid[rounded_potential_new_ball_position]}")
        # we need to bounce, pythagoras to find the incoming angle, then calculate the velocity of the return angle?
        bouncing_object = state.grid[rounded_potential_new_ball_position]
        bounce(state, bouncing_object, log_messages, tick_index)
    else:
        state.position_y, state.position_x = potential_new_y, potential_new_x

    log_messages.append(f"{tick_index}: resulting position is {state.rounded_ball_position}")

    # --- Update the grid ---

    # wipe the previous positions of ball and paddles
    state.grid[np.isin(state.grid, [LEFT_PADDLE, RIGHT_PADDLE, BALL])] = EMPTY_SPACE

    # render the new ones
    state.grid[state.rounded_ball_position] = BALL
    state.grid[state.left_paddle_position] = LEFT_PADDLE
    state.grid[state.right_paddle_position] = RIGHT_PADDLE


def main():
    sys.stdout.write(HIDE_CURSOR + CLEAR_SCREEN)

    state = init_state()
    log_messages = []

    try:
        for i in range(10000):
            time.sleep(1 / 30)
            tick(state, log_messages, i)
            render(state)
    except Exception as error:
        print('\n')
        for log_message in log_messages:
            print(log_message)

        print(error)
    except KeyboardInterrupt:
        print("\nExiting cleanly")
    finally:
        sys.stdout.write(SHOW_CURSOR)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
