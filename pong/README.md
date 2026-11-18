# Pong
Pong is a game for bouncing balls between paddles and walls.

## The Game
- Simple pong game. Can be played single or multiplayer.
- Everyone starts with 3 lives.
- In single player the goal is to score the highest score.
- In multiplater the goal is simple, outlive your opponent.

---

# Single Player:

## The Competitors
There are three different competitors:
- Human (me) - via keyboard interface, fairly limited skills, bad reaction times.
- Scripted - a 'naive' automated approach, matches the y value of the paddle to the y value of the ball.
- Model - A model trained using Reinforcement Learning.

## Testing
- Automatic competitors are tested over 100 games.
- The winner is the competitor with the highest average score.
- The huamn competitor score is just updated manually every time there is a new high score. I am pretty crap.

## The Goal 
- Build a model that can beat the scripted controller.
- The scripted controller is already way better than mee so the scripted controller is the benchmark to beat.

| Controller | Average Score | Min Score | Max Score | Notes               |
|------------|---------------|-----------|-----------|---------------------|
| Scripted   | 556.4         | 140       | 1300      | Pretty bloody good. |
| Human      | 100           | N/a       | N/a       | Embarassing         |

## Notable Model Results
Each new model which achieves a high score is added here and saved in the `pong/models` directory.

| Model Iteration | Average Score | Min Score | Max Score | Notes                                                                                                                                                                                                                              |
|-----------------|---------------|-----------|-----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 0               | 378.7         | 50        | 950       | First proper model trained, 8 vectorised environments, early stopped after 800,000 steps.                                                                                                                                          |
| 6               | 451.6         | 160       | 1050      | Used a linear learning rate, decaying from 0.0003 to 0.00001. Early stopped after 1,000,000 steps.                                                                                                                                 |
| 12              | 500.3         | 140       | 1100      | Started from it.6 weights, then updated training env so game is started from a randomised start point, providing more exposure to higher levels. Same linear learning rate.                                                        |
| 15              | 559.0         | 200       | 1350      | Started from it.6 weights, used same randomised start point as it.12, but reduced random start to 0 - 20 paddle bounces (from 0-50). Early stopping enabled but not used.                                                          |
| 22              | 650.0         | 200       | 1400      | Started from fresh weights, used updated randomised start which should provide more accurate simulation. Trained for 4,000,000 steps, no early stopping but stored best eval. Learning rate linear decayed from 0.0003 to 0.00001. |
| 24              | 1212.4        | 180       | 2400      | Wow, started from fresh weights, trained for 10,000,000 steps, no early stopping but stored best eval. Learning rate linear decayed from 0.0003 to 0.00001 at 0.9.                                                                 |


----

# Multiplayer