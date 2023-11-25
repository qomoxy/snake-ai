import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import os


class Linear_QNet(nn.Module):
    """class Linear_QNet"""

    def __init__(self, input_size, hidden_size, output_size) -> None:
        """
        build the neural network
        :param input_size: input size
        :param hidden_size: hidden size
        :param output_size: output size
        """
        super().__init__() # call the constructor of the parent class nn.Module

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x) -> torch.tensor:
        """
        forward function
        :param x: input
        :return: x
        """
        x = f.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pht') -> None:
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    """class QTrainer"""
    def __init__(self, model, lr, gamma) -> None:
        """build the trainer"""
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done) -> None:
        """
        train the model, update the weights, update the neural network, update the Q-table.
        :param state: state of the game (snake, food, direction)
        :param action: action of the model (left, right, up, down)
        :param reward: reward(s) for the action(s), if the game is over, the reward is -10. If he eats the food, the reward is +10. Otherwise, the reward is 0.
        :param next_state: next state of the model
        :param done: if the game is over or not
        :return: None
        """
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        next_state = torch.tensor(next_state, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            action = torch.unsqueeze(action, 0)
            next_state = torch.unsqueeze(next_state, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        prd = self.model(state)

        target = prd.clone()
        for i in range(len(done)):
            Q_new = reward[i]
            if not done[i]:
                Q_new = reward[i] + self.gamma * torch.max(self.model(next_state[i]))

            target[i][torch.argmax(action).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, prd)
        loss.backward()

        self.optimizer.step()
