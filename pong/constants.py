import math

import numpy as np

RANDOM_SEED = 530

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

LEVEL_EASY_MAX_BOUNCES = 10  # move to level MEDIUM after 10 paddle bounces.
LEVEL_MEDIUM_MAX_BOUNCES = 20
LEVEL_HARD_NEW_BALLS_PADDLE_BOUNCES = 10  # in hard mode every X paddle bounces we add a new ball.

LEVEL_EASY_SCORE_POINTS = 10  # Points gained per bounce during each level
LEVEL_MEDIUM_SCORE_POINTS = 20
LEVEL_HARD_SCORE_POINTS = 50

MAX_BALLS = 5
MAX_BALL_VELOCITY = 0.99

# Physical constants
GRID_SIZE_Y = 10
GRID_SIZE_X = 30

LEFT_PADDLE_X_VALUE = 1
RIGHT_PADDLE_X_VALUE = GRID_SIZE_X - 2

PADDLE_HALF_HEIGHT = 0.5

# Calculate the starting velocity - tie it to grid size so velocity scales with grid, but convert to a scalar so we can randomise the direction when we start.
STARTING_Y_VELOCITY = GRID_SIZE_Y / 100
STARTING_X_VELOCITY = GRID_SIZE_X / 100
STARTING_VELOCITY_SCALAR = math.hypot(STARTING_X_VELOCITY, STARTING_Y_VELOCITY)
MAX_STARTING_ANGLE = math.radians(45)

LEVEL_EASY_BALL_Y_ACCELERATION = STARTING_Y_VELOCITY / 10  # ball accelerates this amount per bounce on a paddle
LEVEL_EASY_BALL_X_ACCELERATION = STARTING_X_VELOCITY / 10
LEVEL_MEDIUM_HIGH_BALL_Y_ACCELERATION = LEVEL_EASY_BALL_Y_ACCELERATION * 2
LEVEL_MEDIUM_HIGH_BALL_X_ACCELERATION = LEVEL_EASY_BALL_X_ACCELERATION * 2

# Maps object to unit normal vector, used in the calculation of velocity after a bounce.
OBJECT_TO_UNIT_NORMAL_VECTOR = {
    DEATH_ZONE_LEFT: np.array((0, 1)),  # Vertical wall (left)
    DEATH_ZONE_RIGHT: np.array((0, -1)),  # Vertical wall (right)
    FLOOR: np.array((1, 0)),  # Horizontal floor
    CEILING: np.array((-1, 0)),  # Horizontal ceiling
    LEFT_PADDLE: np.array((0, 1)),
    RIGHT_PADDLE: np.array((0, -1)),
}

# Training
TRAINING_STEPS = 1_000_000
GAMMA = 0.995  # From online: - A useful rule of thumb is that the agent effectively “sees” about 1 / (1 − γ) ticks ahead  where γ is gamma

MAX_PADDLE_BOUNCES = 1000  # truncate after 100 paddle bounces

# Inference
MODEL_PATH = "pong/models/pong_model_it_0"
