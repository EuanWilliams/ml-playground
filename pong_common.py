import numpy as np
import numpy.typing as npt

Grid = npt.NDArray[np.float32]

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

KEY_W = 119
KEY_S = 115
KEY_UP_ARROW = 259
KEY_DOWN_ARROW = 258
STARTING_LIVES = 3


def init_grid() -> Grid:
    """Initialises the grid with zeros"""

    grid = np.zeros((GRID_SIZE_Y, GRID_SIZE_X), dtype=np.float32)
    grid[:, [0]] = DEATH_ZONE_LEFT  # create death zone
    grid[:, [-1]] = DEATH_ZONE_RIGHT  # create death zone
    grid[[0], :] = CEILING
    grid[[-1], :] = FLOOR

    return grid
