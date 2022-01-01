from math import fabs, log
import numpy as np
import random

from union import Union


class MCell(Union):
    def __init__(self, x, y, wall) -> None:
        Union.__init__(self)
        self.x = x
        self.y = y
        self.wall = wall


class Maze:
    WALL = 1
    EMPTY = 0

    def __init__(self, width, height) -> None:
        self.width = width
        self.height = height
        self.modified = False

    def getRandomEmpty(self):
        return self.empty[random.randint(0, len(self.empty) - 1)]

    def setValxy(self, x, y, v):
        self.maze[y][x] = v

    def getValxy(self, x, y):
        self.maze[y][x]

    def getVal(self, pos):
        return self.maze[pos[1]][pos[0]]

    def setVal(self, pos, v):
        # pos : (x,y)
        self.maze[pos[1]][pos[0]] = v

    def reset(self):
        self.maze = []
        self.empty = []
        self.cells: dict[any, MCell] = {}
        self.walls = []
        self.perm = []
        self.iter = 0
        for y in range(self.height):
            ar = []
            for x in range(self.width):
                if x == 0 or x == self.width-1 or y == 0 or y == self.height - 1:
                    ar.append(Maze.WALL)
                else:
                    ar.append(Maze.EMPTY)
                    self.empty.append((x, y))
            self.maze.append(ar)
        self.floor = (self.width - 1)*(self.height-1)

    def getWallsOnly(self):
        walls = []
        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                if x % 2 == 0 or y % 2 == 0:
                    walls.append(self.getVal((x, y)))
        return walls

    def initMaze(self):
        self.reset()
        # Randomized Kruskal's algorithm

        s = 0
        for x in range(1, self.width - 1):
            for y in range(1, self.height - 1):
                cell = MCell(x, y, True)
                if x % 2 == 0 and y % 2 == 0:
                    # walls.append((x,y))
                    self.setValxy(x, y, Maze.WALL)
                    cell.m = 1
                elif x % 2 == 0 or y % 2 == 0:
                    self.walls.append((x, y))
                    self.setValxy(x, y, Maze.WALL)
                else:
                    cell.wall = False
                s += 1
                self.cells[(x, y)] = cell

        self.perm = np.random.permutation(range(len(self.walls)))
        # print(perm)
        # for i in range(len(self.perm)):
        # self.addWall()

    def rebuildMaze(self):
        self.initMaze()
        while self.breakWalls() != -2:
            continue

    def breakWalls(self):
        if self.iter >= len(self.perm):
            return -2

        wall = self.walls[self.perm[self.iter]]
        ni = get_neibours(self, wall[0], wall[1])
        # print(wall, ni)
        connected = True
        f = -1
        for n in ni:
            ccell = self.cells[n]
            if ccell.wall:
                continue
            if f == -1:
                f = ccell.get_group()
            elif ccell.get_group() != f:
                # print(f, self.cells[n].get_group(), n)
                connected = False
                break
        if connected and random.random() > 0.75:
            connected = False

        if not connected:
            wc = self.cells[wall]
            # if wc.x % 2 == 0 and wc.y % 2 == 0:
            #     print(wc.x, wc.y)

            for n in ni:
                ncell = self.cells[n]
                if ncell.wall:
                    r = random.randint(self.iter+1, len(self.walls))
                    self.perm = np.insert(self.perm, r, len(self.walls))
                    self.walls.append((ncell.x, ncell.y))
                    continue
                wc.add_child(ncell)

            wc.wall = False
            self.setValxy(wall[0], wall[1], Maze.EMPTY)
            self.iter += 1
            self.modified = True
            return wall
        self.iter += 1
        return -1


def get_neibours(maze: Maze, x, y):
    ni = set()
    for i in range(-1, 2, 2):
        if x + i < maze.width - 1 and x + i > 0:
            ni.add((x+i, y))
        if y + i < maze.height - 1 and y + i > 0:
            ni.add((x, y+i))
    return ni


def find_free(maze: Maze):
    pass
