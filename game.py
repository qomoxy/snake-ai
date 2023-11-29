import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init() # initialize all imported pygame modules
font = pygame.font.Font('arial.ttf', 25)


# font = pygame.font.SysFont('arial', 25)
# create a new Font object from a file

class Direction(Enum):
    """The direction of the snake"""
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y') # create a new named tuple object

# rgb colors
WHITE = (255, 255, 255) # score color
RED = (200, 0, 0) # game over and food
BLUE1 = (0, 0, 255) # body of the snake
BLUE2 = (0, 100, 255) # head of the snake
BLACK = (0, 0, 0) # background color of the game

BLOCK_SIZE = 20 # 20 px size of the block
SPEED = 1500 # 1500 ms = 1.5 s speed of the game


class SnakeGameAI:
    """class SnakeGameAI"""

    def __init__(self, w=640, h=480) -> None:
        """build the game"""

        self.w = w
        self.h = h

        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()

        # init game state
        self.reset()

    def reset(self) -> None:
        """reset the game"""

        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_itarition = 0

    def _place_food(self) -> None:
        """place the food"""
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action) -> tuple[int, bool, int]:
        """play one step"""
        # 1. collect user input
        self.frame_itarition += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move
        self._move(action)  # update the head
        self.snake.insert(0, self.head)

        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_itarition > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score

    def is_collision(self, point=None) -> bool:
        """
        check if the snake hits itself or the boundary
        :param point: the point to check
        """
        if point is None:
            point = self.head
        # hits boundary
        if point.x > self.w - BLOCK_SIZE or point.x < 0 or point.y > self.h - BLOCK_SIZE or point.y < 0:
            return True
        # hits itself
        if point in self.snake[1:]:
            return True

        return False

    def _update_ui(self) -> None:
        """update the ui"""
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action) -> None:
        """
        move the snake
        :param action: the action to take
        """

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            new_idx = (idx + 1) % 4
            new_dir = clock_wise[new_idx] # right turn
        else : # [0, 0, 1]
            new_idx = (idx - 1) % 4
            new_dir = clock_wise[new_idx]

        self.direction = new_dir

        x = self.head.x
        y = self.head.y

        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
