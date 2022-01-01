import time
import os

from cv2 import log
from model import preprocess_image, combine_states
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from collections import deque
import random
from time import sleep
import pygame
from level import Level
from maze import Maze
from player import Player, Direction
from enemy import Enemy
from mazeToInput import empty_spot, maze_state_to_input, maze_to_1D
from engine import Engine
import numpy as np
import cv2

# generate level object


def preprocess_image(im, image_size=64):
    # print(np.min(im), np.max(im), np.average(im))
    im = cv2.resize(im, (image_size, image_size)) / 255.0
    return im


class Game:
    def __init__(self, maze_width, maze_height, num_enemies, num_foods, fps) -> None:
        self.maze_width = maze_width
        self.maze_height = maze_height
        self.num_enemies = num_enemies
        self.num_foods = num_foods
        self.width = maze_width * 16
        self.height = maze_height * 16
        self.game = Engine(self.width, self.height, fps)
        self.m = Maze(maze_width, maze_height)
        self.level = Level(self.m.width * 16, self.m.height * 16,
                           (255, 255, 224), (222, 184, 135))
        self.input_loop = None
        self.hightlight = False
        self.episode = 0

        self.wall_image = np.ndarray([self.width, self.height])
        self.quit = False
        # self.object_image = np.ndarray([self.width, self.height])
        # self.init()

    def init(self):
        if(self.quit):
            self.game = Engine(self.width, self.height, self.game.fps)
            self.quit = False
        self.m.rebuildMaze()
        self.rebuildLevel()

        screen = self.game.window
        screen.fill((0, 0, 0))
        self.level.draw_black_background(screen, (155, 50, 55))
        self.wall_image = pygame.surfarray.array3d(self.game.window)

        self.ended = False
        self.episode += 1
        self.score = 0
        self.hit = 0
        self.foods = []
        self.food_pos = {}

        # player_pos = empty_spot(self.m.width, self.m.height)
        # while player_pos[0] == 5 and player_pos[0] == 5:
        #     player_pos = empty_spot(self.m.width, self.m.height)
        player_pos = [1, 1]
        if np.random.random() > 0.5:
            player_pos[0] = self.m.width - 2
        if np.random.random() > 0.5:
            player_pos[1] = self.m.height - 2
        self.frame = 0
        self.player = Player(
            self.level, player_pos[0], player_pos[1], 16, 16, 2, 2)
        self.enemies = []

        for i in range(self.num_enemies):
            # enemy_pos = empty_spot(self.m.width, self.m.height)
            # while enemy_pos[0] == player_pos[0] and enemy_pos[1] == player_pos[1]:
            #     enemy_pos = empty_spot(self.m.width, self.m.height)
            enemy_pos = [3, 3]
            if np.random.random() > 0.5:
                enemy_pos[0] = 5
            if np.random.random() > 0.5:
                enemy_pos[1] = 5
            enemy = Enemy(
                self.level, enemy_pos[0], enemy_pos[1], 16, 16, 1.75, 1)
            enemy.move_dir = 0
            self.enemies.append(enemy)

        for i in range(self.num_foods):
            self.food_pos[i] = self.random_food_pos()
            # self.food_pos = (5, 5)
            food = pygame.Rect(
                self.food_pos[i][0] * 16 + 1, self.food_pos[i][1] * 16 + 1, 14, 14)
            self.foods.append(food)

    def has_ended(self):
        return self.ended

    def random_food_pos(self):
        pos = [0, 0]
        stop = False
        while True:
            pos = [random.randint(1, self.maze_width - 2),
                   random.randint(1, self.maze_height - 2)]
            if self.m.getVal(pos) == Maze.WALL:
                continue
            if Direction.same(pos, self.player.posNormalized()):
                continue

            stop = True
            for food_pos in self.food_pos.values():
                if Direction.same(pos, food_pos):
                    stop = False
            if stop:
                break
        return pos

    def state(self):
        # current = np.zeros((self.width - 2, self.height - 2, 3))
        # for x in range(1, self.width):
        #     for y in range(1, self.height):
        #         if
        # return maze_state_to_input(self.m, self.player.posNormalized(), self.food_pos, self.enemies)
        # imgdata = pygame.surfarray.array3d(self.game.window)
        screen = self.game.window
        # screen.fill(self.level.background.color)
        screen.fill((0, 0, 0))
        for food in self.foods:
            pygame.draw.rect(screen, (1, 255, 1), food)

        if self.hightlight:
            pygame.draw.rect(screen, (200, 50, 50), self.player.rect)
        else:
            pygame.draw.rect(screen, (70, 130, 180), self.player.rect)

        for enemy in self.enemies:
            enemy: Enemy
            # print(enemy.rect.center)
            pygame.draw.rect(screen, (255, 50, 50), enemy.rect)

        imgdata = pygame.surfarray.array3d(self.game.window)
        return preprocess_image(imgdata + self.wall_image)

    def eat_food(self):
        for food_rect in self.foods:
            if self.player.rect.colliderect(food_rect):
                food_pos = self.random_food_pos()
                food_rect.x = food_pos[0] * 16 + 1
                food_rect.y = food_pos[1] * 16 + 1
                self.score += 10
                return 10
        return 0

    def step(self, action):
        self.game.update()
        self.player.moveDir(action)
        while self.player.moving():
            # self.game.update()
            self.player.update()
            if self.eat_food():
                return self.state(), 1, False
            for enemy in self.enemies:
                enemy: Enemy
                enemy.enemy_move(self.m)
                enemy.update()
                if self.player.rect.colliderect(enemy.rect):
                    # print("you lost!")
                    self.ended = True
                    return self.state(), -0.25, True
            self.frame += 1
        if self.player.hit_wall:
            self.hit = 0
            self.player.hit_wall = False
            return self.state(), -0.01, False
        self.hit += 1
        if self.hit > 2:
            return self.state(), 0, False
        return self.state(), -0.005, False

    def visualize(self):
        screen = self.game.window
        screen.fill(self.level.background.color)
        self.level.draw(screen)
        for food in self.foods:
            pygame.draw.rect(screen, (0, 255, 0), food)
        if self.hightlight:
            pygame.draw.rect(screen, (200, 50, 50), self.player.rect)
        else:
            pygame.draw.rect(screen, (70, 130, 180), self.player.rect)

        for enemy in self.enemies:
            enemy: Enemy
            # print(enemy.rect.center)
            pygame.draw.rect(screen, (255, 50, 50), enemy.rect)
        pygame.display.flip()

    def rebuildLevel(self):
        self.level.reset()
        x = y = 0
        for y in range(self.m.height):
            for x in range(self.m.width):
                if self.m.maze[y][x] == 1:
                    self.level.addWall(x * 16, y * 16)

    def start(self, input_loop):
        self.input_loop = input_loop
        self.game.init(self)

    def close(self):
        self.quit = True
        pygame.quit()

    def loop(self, screen: pygame.Surface):
        key = self.input_loop()
        if(key == Direction.SKIP):
            self.init()
            return True
        self.player.moveDir(key)
        # player.move(0, 1)
        # m.addWall()
        # rebuildLevel(level)
        screen.fill(self.level.background.color)
        self.level.draw(screen)
        if self.hightlight:
            pygame.draw.rect(screen, (200, 50, 50), self.player.rect)
        else:
            pygame.draw.rect(screen, (70, 130, 180), self.player.rect)

        self.player.update()
        if self.eat_food():
            print("score :", self.score)

        for food in self.foods:
            pygame.draw.rect(screen, (0, 255, 0), food)

        for enemy in self.enemies:
            enemy: Enemy
            enemy.enemy_move(self.m)
            enemy.update()
            pygame.draw.rect(screen, (255, 50, 50), enemy.rect)
            if self.player.rect.colliderect(enemy.rect):
                print("you lost!")
                self.init()
        self.frame += 1

        return True


def input_loop_human():
    key = pygame.key.get_pressed()
    if key[pygame.K_LEFT]:
        return Direction.LEFT
        # player.move(-1, 0)
    if key[pygame.K_RIGHT]:
        return Direction.RIGHT
        # player.move(1, 0)
    if key[pygame.K_UP]:
        return Direction.UP
        # player.move(0, -1)
    if key[pygame.K_DOWN]:
        return Direction.DOWN
    if key[pygame.K_s]:
        return Direction.SKIP
    if key[pygame.K_q]:
        return Direction.STOP
    if key[pygame.K_n]:
        return Direction.NEXT
    return Direction.NONE


def run_human():
    game = Game(640, 640, 30)
    print(game.state(), game.state().shape)
    print(game.m.maze)
    game.start(input_loop_human)
    exit()


# Declaring deque
# arr = deque(maxlen=5)

# for i in range(15):
#     arr.append(i)
#     print(arr)

# arr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# print(arr[-4:])
# run_human()


# game = Game(30)
# game.init()
# game.visualize()


# start = time.time()
# for i in range(5000):
#     imgdata = game.state()
# end = time.time()
# print("end of p1")
# for i in range(5000):
#     # game.visualize()
#     imgdata2 = pygame.surfarray.array3d(game.game.window)
# end2 = time.time()

# print(end - start, end2 - end)
# # imgdata = pygame.surfarray.array3d(game.game.window)

# imgdata = game.state()


# plt.imshow(imgdata)
# plt.show()

# q = deque(maxlen=4)
# q.append(imgdata)
# q.append(imgdata)
# q.append(imgdata)
# q.append(imgdata)

# imgdata = combine_states(q)
# print(imgdata.shape)
# print(imgdata[:, :, :3].shape)
# plt.imshow(imgdata[:, :, :3])
# plt.show()

# arr = np.random.rand(9, 9, 3)
# print(arr)

# arr = np.concatenate([arr, arr, arr, arr], axis=2)
# print(arr.shape)
