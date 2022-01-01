from math import log
import math
from maze import Maze
from enemy import Enemy
import numpy as np


def maze_state_to_input(maze: Maze, pos_player, pos_food, post_enemies):
    data = np.array(maze.maze)
    data = np.dstack((data, data, data))
    data[pos_player[1], pos_player[0], 2] = 1
    data[pos_food[0], pos_food[1], 1] = 1
    for enemy in post_enemies:
        enemy: Enemy
        norm = enemy.posNormalized()
        data[norm[1], norm[0], 0] = 1
    return data


def maze_to_1D(maze: Maze):
    # print(np.array(maze.maze))
    fm = np.array(maze.maze)[1:-1, 1:-1]
    #fm = np.array(maze.getWallsOnly())
    # print(fm)
    # print(maze.maze)
    # std = 0.5 * (fm.shape[0] + fm.shape[0]) / 2
    # print(fm.shape, 29, std)
    return (fm.flatten() * 2) - 1


def input_add_pos(X, pos, width, height):
    n = (width - 2)*(height - 2)
    avg = 1/n
    std = (1/n)*(1-1/n)
    std = math.sqrt(std)

    return add_pos_std_avg(X, pos, width, height, avg, std)


def add_pos_std_avg(X, pos, width, height, avg, std):
    arr = np.zeros([width - 2, height - 2])
    arr[pos[0]-1][pos[1]-1] = 1
    arr = (arr - avg) / std
    arr = arr.flatten()

    return np.append(X, arr)


def empty_spot(width, height):
    x = np.random.randint(0, np.floor((width) / 2)) * 2 + 1
    y = np.random.randint(0, np.floor((height) / 2)) * 2 + 1
    return x, y


def getRandom(X, Y, indices, samples):
    perm = np.random.permutation(indices)
    return X[perm][0:samples], Y[perm][0:samples]
