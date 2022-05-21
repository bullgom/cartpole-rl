import torch
import torch.nn as NN
import torch.nn.functional as F
import numpy as np
import typing
from dataclasses import dataclass
from torchvision.transforms import transforms
import random

Action = typing.Any


class EpsilonGreedyParameters:

    # Probability of selecting randomly
    e_start:  float
    e_end: float
    e_decrease_episode: int

    def __init__(self, e_start: float, e_end: float, episodes: int) -> None:
        self.e_start = e_start
        self.e_end = e_end
        self.episodes = episodes
        self.e = self.e_start
        self.rate = (self.e_start - self.e_end) / self.episodes

    def step(self):
        self.e -= self.rate
        if self.e < self.e_end:
            self.e = self.e_end


class MyLovelyAgent(torch.nn.Module):
    """
    This agent acts on CartPole with image as input
    Learns to approaximate the value function by using Semi-gradient TD(0)
    introduced in Reinforcement Learning An Intro. 2nd Ed. by Richard S. Sutton
    page 203
    """

    def __init__(
        self,
        image_size: typing.Union[int, tuple[int, int]],
        action_set: set[Action],
        e_greedy_parameters: EpsilonGreedyParameters,
        device: torch.device
    ) -> None:

        super().__init__()
        self.input_shape = image_size
        self.action_set = action_set
        self.e_greedy_parameters = e_greedy_parameters
        self.device = device

        self.init_layers(action_set)
        self.to(device)

    def init_layers(self, action_set: set[Action]):
        """Define layers here"""
        if type(self.input_shape) is int:
            input_width, input_height = self.input_shape, self.input_shape
        else:
            input_width, input_height = self.input_shape

        self.conv1 = NN.Conv2d(1, 16, kernel_size=5, stride=2)
        self.bn1 = NN.BatchNorm2d(16)
        self.conv2 = NN.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = NN.BatchNorm2d(32)
        self.conv3 = NN.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = NN.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_width)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_height)))
        linear_input_size = convw * convh * 32
        self.head = NN.Linear(linear_input_size, len(action_set))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Receive an image as tensor output the estimated value of each action
        TODO: Also define activations in `init_layers` function
        """
        assert issubclass(type(x), torch.Tensor)
        x = x.to(self.device)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

    def make_up_my_mind(self, image: torch.Tensor) -> Action:
        """
        Select action using e-greedy policy on values from the
        value approaximator function a.k.a DQN. Is the naming right?
        TODO: Make this general across multiple policies
        """
        # we'll be acting on the real world, so its a test! Put it in test mode

        non_random = np.random.uniform() > self.e_greedy_parameters.e
        if non_random:  # Select Max
            with torch.no_grad():
                action_index = self(image).max(1)[1].view(1, 1)
        else:  # Select Randomly
            action_index = torch.tensor(
                [[random.randrange(len(self.action_set))]], device=self.device, dtype=torch.long)

        return action_index.to(self.device)

    def copy(self) -> "MyLovelyAgent":
        new_agent = MyLovelyAgent(self.input_shape, self.action_set,
                                  self.e_greedy_parameters, self.device)
        new_agent.load_state_dict(self.state_dict())

        return new_agent
