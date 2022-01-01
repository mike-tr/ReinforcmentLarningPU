from gym.spaces import Discrete, Box
from gym import Env
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.agents import DQNAgent
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Convolution2D
from tensorflow.keras.models import Sequential
import random
import time
from game import Game
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time

print(tf.__version__)

env = Game(9, 9, 2, 5, 30)
env.init()

height, width, channels = env.state().shape
actions = 4

print(height, width, channels)

#imgdata = env.state()
# plt.imshow(imgdata)
# print(np.max(imgdata), np.min(imgdata), np.average(imgdata))
# plt.show()
env.close()


class GameEnv(Env):
    def __init__(self, maze_width, maze_height, num_enemies, num_foods):
        # Actions we can take, left, right, up, down
        self.start_time = 150
        self.action_space = Discrete(4)

        self.game = Game(maze_width, maze_height, num_enemies, num_foods, 30)
        self.game.init()

        shape = self.game.state().shape
        # picture array
        self.observation_space = Box(low=np.zeros(shape), high=np.ones(shape))
        # Set start temp
        self.state = self.game.state()
        # Game length
        self.time_left = self.start_time

    def step(self, action):
        self.state, reward, done = self.game.step(action)
        self.time_left -= 1
        # Check if shower is done
        if self.time_left <= 0:
            done = True
        elif not done:
            done = False

        # Apply temperature noise
        #self.state += random.randint(-1,1)
        # Set placeholder for info
        info = {}

        # Return step information
        return self.state, reward, done, info

    def render(self, mode=None):
        time.sleep(0.05)
        self.game.visualize()

    def reset(self):
        self.game.init()
        self.state = self.game.state()
        self.time_left = self.start_time
        return self.state

    def close(self):
        self.game.close()


def build_model(height, width, channels, actions):
    model = Sequential()
    model.add(Convolution2D(32, (5, 5), strides=(2, 2),
              activation='relu', input_shape=(3, height, width, channels)))
    model.add(Convolution2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model


def build_agent(model, actions):
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(
    ), attr='eps', value_max=1., value_min=.1, value_test=0, nb_steps=75000)
    memory = SequentialMemory(limit=1000, window_length=3)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                   enable_dueling_network=True, dueling_type='avg',
                   nb_actions=actions, nb_steps_warmup=1000)
    return dqn


def load_model(modelName):
    model = build_model(height, width, channels, 4)
    dqn = build_agent(model, 4)
    dqn.compile(Adam(learning_rate=1e-4))
    dqn.load_weights('models/' + modelName + '.h5f')
    return dqn


def visualize_test(model, size, enemies, food):
    env = GameEnv(size, size, enemies, food)
    scores = model.test(env, nb_episodes=10, visualize=True)
    print(np.mean(scores.history['episode_reward']))
    env.close()


def trainModel(modelName, episodes, size, enemies, food, modelOld=None):
    env = GameEnv(size, size, enemies, food)

    model = build_model(height, width, channels, 4)
    model.summary()

    dqn = build_agent(model, 4)
    dqn.compile(Adam(learning_rate=1e-4))

    if modelOld != None:
        dqn.load_weights('models/' + modelOld + '.h5f')

    dqn.fit(env, nb_steps=episodes, visualize=False, verbose=2)
    dqn.save_weights('models/' + modelName + '.h5f')

    env.close()


oldModel = '/dqn/50k/dqn_weight'
newModel = '/dqn/150k/dqn_weight'
newModel2 = '/dqn/250k/dqn_weight'
newModel3 = '/dqn/350k/dqn_weight'
newModel4 = '/dqn/450k/dqn_weight'
# trainModel(newModel, 100000, oldModel)
trainModel(newModel2, 100000, 15, 5, 8, newModel)
# trainModel(newModel3, 100000, newModel2)
# trainModel(newModel4, 100000, newModel3)

model10k = '/dqn/50k/dqn_weight_n'
model50k = '/dqn/50k/dqn_weight'
model150k = '/dqn/50k/dqn_weight'


# model = load_model(model10k)
# visualize_test(model)
# model = load_model(model50k)
# visualize_test(model)

# # original
# model = load_model(model150k)
# visualize_test(model, 9, 2, 5)
#visualize_test(model, 11, 3, 5)
# visualize_test(model, 15, 5, 8)
