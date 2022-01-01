import pygame
from pygame import math
import math


class Direction:
    LEFT = 0
    RIGHT = 1
    UP = 3
    DOWN = 2
    NONE = -1
    SKIP = -2
    STOP = -3
    NEXT = -4

    def getVec(direction):
        if direction == Direction.LEFT:
            return (-1, 0)
        elif direction == Direction.RIGHT:
            return (1, 0)
        elif direction == Direction.UP:
            return (0, -1)
        elif direction == Direction.DOWN:
            return (0, 1)

    def AddPos(pos, direction):
        vec = Direction.getVec(direction)
        return pos[0] + vec[0], pos[1] + vec[1]

    def valid(direction):
        if 0 <= direction <= 4:
            return True

    def same(vec1, vec2):
        return vec1[0] == vec2[0] and vec1[1] == vec2[1]

    def to_string(direction):
        if direction == Direction.LEFT:
            return "Left"
        if direction == Direction.RIGHT:
            return "RIGHT"
        if direction == Direction.UP:
            return "UP"
        if direction == Direction.DOWN:
            return "DOWN"


class Player(object):

    def __init__(self, level, x, y, width, height, size, speed):
        self.speed = speed
        self.level = level
        self.width = width
        self.height = height
        self.rect = pygame.Rect(x * width + width/(size * 2),
                                y * height + height/(size * 2), width/size, height/size)
        self.direction = Direction.NONE
        self.ticks_left = 0
        self.ticks_per_move = width / speed
        self.hit_wall = False
        # print(self.ticks_per_move)

    def posNormalized(self):
        return math.floor(self.rect.centerx / self.width), math.floor(self.rect.centery / self.height)

    def pos(self):
        return self.rect.center

    def update(self):
        # add velocity until in the correct position
        # print(self.direction, self.ticks_left, self.ticks_per_move)
        if self.direction != -1:
            if self.direction == Direction.LEFT:
                self.move(-self.speed, 0)
            if self.direction == Direction.RIGHT:
                self.move(self.speed, 0)
            if self.direction == Direction.UP:
                self.move(0, -self.speed)
            if self.direction == Direction.DOWN:
                self.move(0, self.speed)
            self.ticks_left -= 1
            if self.ticks_left <= 0:
                self.direction = Direction.NONE

    def moveDir(self, direction):
        if self.direction != Direction.NONE:
            return
        if Direction.valid(direction):
            self.direction = direction
            self.ticks_left = self.ticks_per_move

    def move(self, dx, dy):
        # Move each axis separately. Note that this checks for collisions both times.
        if dx != 0:
            self.move_single_axis(dx, 0)
        if dy != 0:
            self.move_single_axis(0, dy)

    def moving(self):
        return self.direction != Direction.NONE

    def move_single_axis(self, dx, dy):

        # Move the rect
        self.rect.x += dx
        self.rect.y += dy

        # If you collide with a wall, move out based on velocity
        for wall in self.level.walls:
            if self.rect.colliderect(wall.rect):
                if dx > 0:  # Moving right; Hit the left side of the wall
                    self.rect.right = wall.rect.left
                if dx < 0:  # Moving left; Hit the right side of the wall
                    self.rect.left = wall.rect.right
                if dy > 0:  # Moving down; Hit the top side of the wall
                    self.rect.bottom = wall.rect.top
                if dy < 0:  # Moving up; Hit the bottom side of the wall
                    self.rect.top = wall.rect.bottom
                self.hit_wall = True
                #self.direction = Direction.NONE
