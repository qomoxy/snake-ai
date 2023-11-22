import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class Linear_QNet(nn.Module):
    """class Linear_QNet"""

    def __init__(self, input_size : int, hidden_size : int, output_size : int):
        """
        build the neural network
        :param input_size: input size
        :param hidden_size: hidden size
        :param output_size: output size
        """
        super().__init__()

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        forward function
        :param x: input
        :return: x
        """
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pht'):
        pass
