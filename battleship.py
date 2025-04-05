import gym
import gym_battleship

env = gym.make('Battleship-v0', ship_sizes={5: 1, 4: 1, 3: 2, 2: 1}, board_size=(10, 10))
env.reset()

ACTION_SPACE = env.action_space.n
OBSERVATION_SPACE = env.observation_space.shape[0]

for i in range(10):
    env.step(env.action_space.sample())

print(env.board_generated)
env.render_board_generated()