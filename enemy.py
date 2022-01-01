import pygame
from pygame import math
import math

from player import Direction, Player
from maze import Maze
import numpy as np


class Enemy(Player):
    def __init__(self, level, x, y, width, height, size, speed):
        super().__init__(level, x, y, width, height, size, speed)
        self.move_dir = Direction.RIGHT

    def enemy_move(self, maze: Maze):
        if self.direction == Direction.NONE:
            valid_moves = self.validMoves(maze)
            # print(valid_moves)
            if self.move_dir not in valid_moves or np.random.random() > 0.5:
                index = np.random.randint(0, len(valid_moves))
                self.move_dir = valid_moves[index]
            # print(Direction.to_string(self.move_dir))
        self.moveDir(self.move_dir)

    def validMoves(self, maze: Maze):
        moves = []
        pos = self.posNormalized()
        for direction in [Direction.LEFT, Direction.RIGHT, Direction.UP, Direction.DOWN]:
            if maze.getVal(Direction.AddPos(pos, direction)) == Maze.EMPTY:
                moves.append(direction)
        return moves
