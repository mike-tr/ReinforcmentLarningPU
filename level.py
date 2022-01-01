import pygame


class Wall(object):
    def __init__(self, level, x, y, width, height):
        level.walls.append(self)
        self.rect = pygame.Rect(x, y, width, height)


class Block(object):
    def __init__(self, x, y, width, height, r, g, b):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = (r, g, b)


class Level:
    def __init__(self, width, height, background_color, wall_color) -> None:
        self.walls: list[Wall] = []
        self.path = []
        self.background = Block(
            0, 0, width, height, background_color[0], background_color[1], background_color[2])
        self.wall_color = wall_color

    def reset(self):
        self.walls: list[Wall] = []

    def addWall(self, x, y):
        Wall(self, x, y, 16, 16)

    def add_path(self, path):
        if path == None:
            return
        self.path = []
        for pos in path:
            self.addBlock(pos[0] * 16, pos[1] * 16)

    def addBlock(self, x, y):
        self.path.append(Block(x, y, 16, 16, 200, 0, 0))

    def draw_black_background(self, screen, wall_color):
        for wall in self.walls:
            wall: Wall
            pygame.draw.rect(screen, wall_color, wall.rect)

    def draw(self, screen):
        # pygame.draw.rect(screen, self.background.color, self.background.rect)
        i = 0
        for pnode in self.path:
            if i == 0:
                i += 1
                pygame.draw.rect(screen, pnode.color, pnode.rect)

        for wall in self.walls:
            wall: Wall
            pygame.draw.rect(screen, self.wall_color, wall.rect)
