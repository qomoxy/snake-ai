import random as rn
import numpy as np
import torch
from collections import deque

from game import SnakeGameAI, Direction, Point
from model import QTrainer, Linear_QNet
from helper import plot

MAX_MEMORY = 100_000 # max memory size
BATCH_SIZE = 1000 # mini batch size
LR = 0.001 # learning rate


class Agent:

    def __init__(self) -> None:
        """
        build the agent
        """
        self.nb_game = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game) -> np.array:
        """
        get the state of the game
        :param game: instance of the game
        """
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),
            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),
            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done) -> None:
        """
        remember the state of the game
        :param state: state of the game
        :param action: action of the model (left, right, straight)
        :param reward: the reward of the game (score)
        :param next_state: the next state of the game
        :param done: if the game is done
        :return: None
        """
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self) -> None:
        """
        train the long memory
        :return: None
        """
        if len(self.memory) > BATCH_SIZE:
            mini_sample = rn.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done) -> None:
        """
        train the short memory
        :param state: state of the game
        :param action: action of the model (left, right, straight)
        :param reward: the reward of the game (score)
        :param next_state: the next state of the game
        :param done: if the game is done
        :return: None
        """
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state) -> list[int]:
        """
        get the action of the model
        :param state: state of the game
        :return: the action of the model
        """
        self.epsilon = 80 - self.nb_game
        final_move = [0, 0, 0]
        if rn.randint(0, 200) < self.epsilon:
            move = rn.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train() -> None:
    """
    train the model
    :return: None
    """
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    
    while True:
        
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.nb_game += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print(f'Game {agent.nb_game} Score : {score} Record : {record}') # print stats

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.nb_game
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    '''Do the training'''
    train()
