import pygame


class Draw:
    def loop(self, screen: pygame.Surface):
        # do something each frame
        # retun false, if wish to close game.
        return True


class Engine:
    def __init__(self, width, length, fps) -> None:
        pygame.init()
        self.window = pygame.display.set_mode((width, length))
        self.run = True
        self.draw: Draw = None
        self.clock = pygame.time.Clock()
        self.fps = fps

    def init(self, draw):
        self.draw = draw
        self.loop()

    def update(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.run = False
        if not self.run:
            pygame.quit()
            exit()

    def loop(self):
        while self.run:
            self.clock.tick(self.fps)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.run = False
            self.run = self.draw.loop(self.window) and self.run
            pygame.display.flip()
        pygame.quit()
        exit()
