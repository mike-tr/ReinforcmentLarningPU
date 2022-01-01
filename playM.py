from maze import Maze
from game import Game, input_loop_human
from model import create_maze_solving_network, add_rl_loss_to_network, predict_on_model, combine_states, transfer_weights_partially
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras as keras
from collections import deque
import random
from dataclasses import dataclass
from typing import Any
import time
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from player import Direction

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# model.train(train_x, train_y, 40001, 200, 1000)

width = height = 11*16


@dataclass
class SingleStep:
    state: Any
    next_state: Any
    action: int
    reward: float
    done: bool


def run_episode(game: Game, model, eps, memory, verbose=False, max_steps=None):
    # if not memory:
    #     memory = []
    game.init()
    states = []
    states.append(game.state())
    states.append(game.state())
    states.append(game.state())
    states.append(game.state())
    # m.randomize_agent()
    final_score = 0

    itr = 0
    agents = []

    # state = combine_states(states[-4:])
    # next_state = combine_states(states[-4:])
    while not game.has_ended():  # and not m.has_died():
        state = combine_states(states[-4:])
        itr += 1
        if max_steps and itr > max_steps:
            return final_score
        # if random.random() > anneal_probability(i, max_episodes, switch_episodes, 0.5) or i < switch_episodes:
        if random.random() < eps:
            action = random.randint(0, 3)
            # print(state, state.shape)
        else:
            action = predict_on_model(state, model, False)

        rt, _ = game.step(action)
        # print(game.state().shape)
        # print(np.sum(game.state()))
        # # plt.imshow(game.state())
        # # plt.show()
        states.append(game.state())
        next_state = combine_states(states[-4:])
        final_score += rt

        # if verbose:
        #     m.visualize()
        #     time.sleep(0.05)

        done = final_score if game.has_ended() else 0
        memory.append(SingleStep(
            state=state, next_state=next_state, reward=rt, action=action, done=done))
    # print(f"finished episode with final score of {final_score} and in {itr} iterations")
    return final_score


def main(experiment_name, fw, starting_weights=None):
    # side_len = 5
    folder = f'models/{experiment_name}'
    if not os.path.exists(folder):
        os.makedirs(folder)
    g = 0.95
    mem_size = 5000
    max_frames = 250
    batch_size = 32
    # memory = deque(maxlen=1000)

    if not starting_weights:
        model = create_maze_solving_network()
        target_model = create_maze_solving_network()
        target_model.set_weights(model.get_weights())
        starting_episode = 0
    else:
        model = tf.keras.models.load_model(f'{folder}/{starting_weights}.h5')
        target_model = tf.keras.models.load_model(
            f'{folder}/{starting_weights}.h5')
        starting_episode = int(os.path.splitext(
            os.path.basename(f'{folder}/{starting_weights}.h5'))[0])

    train_model = add_rl_loss_to_network(model)
    eps = 1.0
    decay_factor = 0.9999

    memory = deque(maxlen=mem_size)

    def tbwrite(name, data, step):
        tf.summary.scalar(name, data=data, step=step)

    print("bootstrapping", eps)
    while len(memory) < 2000:
        print(len(memory))
        m = Game(60)
        run_episode(m, target_model, eps, memory, False, max_steps=max_frames)
    print("done bootstrapping")

    rewards = 0
    current_reward = 0

    for i in range(starting_episode, 1000000):
        m = Game(60)
        current_reward += run_episode(m, model,
                                      eps, memory, False, max_steps=max_frames)
        rewards += 1

        steps = random.sample(memory, min(batch_size, len(memory)))

        inputs = []
        outputs = []
        masks = []

        s: SingleStep
        current_actions = model.predict(
            np.stack([s.state for s in steps], 0))
        future_actions = target_model.predict(
            np.stack([s.next_state for s in steps], 0))

        for j, s in enumerate(steps):
            x = s.state
            reward = s.reward
            # target_vector = predict_on_model(s.st, model, True)
            # fut_action = predict_on_model(s.stn, target_model, True)
            target_q, future_q = current_actions[j].copy(
            ), future_actions[j].copy()
            action_target_q = reward
            if not s.done:
                action_target_q += g * np.max(future_q)
            target_q[s.action] = action_target_q
            # put 1 at the right place and 0's in all other places
            mask = target_q.copy()*0
            mask[s.action] = 1

            inputs.append(x)
            outputs.append(target_q)
            masks.append(mask)

        # model.fit(np.stack(inputs, 0), np.stack(outputs, 0), epochs=1)
        inputs = np.stack(inputs, 0)
        targets = np.stack(outputs, 0)
        masks = np.stack(masks, 0)
        if i % 50 == 0:
            hist = train_model.fit([inputs, targets, masks],
                                   targets, epochs=1)
            print("mean reward : ", current_reward / rewards, 'episode :', i)
        else:
            hist = train_model.fit([inputs, targets, masks],
                                   targets, epochs=1, verbose=0)
        # print(hist)
        tbwrite('eps', eps, i)
        eps *= decay_factor
        eps = max(eps, 0.1)
        for k, v in hist.history.items():
            tbwrite(k, v[0], i)

        if i % 100 == 0:
            m = Game(60)
            m.init()
            transfer_weights_partially(model, target_model, 1)
            target_model.save(f'{folder}/{i}.h5')

            frames = []
            frames.append(m.state())
            frames.append(m.state())
            frames.append(m.state())
            frames.append(m.state())
            m.visualize()
            idx = 0
            score = 0
            while not m.has_ended():
                time.sleep(0.1)
                state = combine_states(frames[-4:])
                _score, _ = m.step(predict_on_model(
                    state, target_model, False))
                frames.append(m.state())
                score += _score
                m.visualize()
                idx += 1
                if idx > 50:
                    break
            tbwrite('test_score', score, i)
            # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
            print('test score', score, i, "moves :", idx)
            dones = [m.done for m in memory if m.done][-100:]
            print('mean_score', sum(dones)/len(dones), i)
            # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        dones = [m.done for m in memory if m.done][-100:]
        tbwrite('mean_score', sum(dones)/len(dones), i)


# run_comp()

def run_comp(experiment_name, model_name):
    game = Game(30)
    # print(state.shape)
    # print()
    # print(model.predict_class(train_x[0:25]), train_y[0:25])
    memory = []
    global ep
    ep = 0

    folder = f'models/{experiment_name}'
    if not os.path.exists(folder):
        print("Err folder doesn't exist")
        return
    # memory = deque(maxlen=1000)

    model = tf.keras.models.load_model(f'{folder}/{model_name}.h5')

    def input_loop():
        global ep
        if game.episode > ep:
            ep = game.episode
            memory.append(game.state())
            memory.append(game.state())
            memory.append(game.state())
        if game.frame % 10 == 0:
            memory.append(game.state())
        state = combine_states(memory[-4:])
        print(state[:, :, 0])
        action = predict_on_model(state, model, False)

        human_action = input_loop_human()
        game.hightlight = False
        if human_action == Direction.NONE:
            return action
        game.hightlight = True

        return human_action
    game.start(input_loop)
    exit()


def run_comp2(experiment_name, model_name):
    game = Game(30)
    # print(state.shape)
    # print()
    # print(model.predict_class(train_x[0:25]), train_y[0:25])
    memory = []
    global ep
    ep = 0

    folder = f'models/{experiment_name}'
    if not os.path.exists(folder):
        print("Err folder doesn't exist")
        return
    # memory = deque(maxlen=1000)

    model = tf.keras.models.load_model(f'{folder}/{model_name}.h5')

    memory.append(game.state())
    memory.append(game.state())
    memory.append(game.state())
    memory.append(game.state())

    while not game.has_ended():
        game.visualize()
        # print("start")
        time.sleep(0.1)
        state = combine_states(memory[-4:])
        # print(state[:, :, 0])
        action = predict_on_model(state, model, False)
        game.step(action)
        memory.append(game.state())
    exit()


def run_training(experiment_name, starting_weights=None):
    logdir = "logs/scalars/" + experiment_name
    file_writer = tf.summary.create_file_writer(logdir + "/metrics")
    file_writer.set_as_default()
    main(experiment_name, file_writer, starting_weights)


if __name__ == '__main__':
    # argh.dispatch_commands([run_training, run_test])
    model_name = None
    experiment_name = "QlearningModel8"
    # run_training(experiment_name, model_name)
    run_comp2("AtariModel3", "100000")
