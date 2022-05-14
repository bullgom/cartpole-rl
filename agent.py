import torch
import torch.nn as NN
import torch.nn.functional as F
import numpy as np
import typing
from dataclasses import dataclass
from torchvision.transforms import transforms

Action = typing.Any


@dataclass
class EpsilonGreedyParameters:

    # Probability of selecting randomly
    e: float

    @property
    def e(self) -> float:
        return self._e

    @e.setter
    def e(self, value: float):
        if not issubclass(type(value), float):
            raise TypeError(f"{type(value)} is not float!")
        if not (0 <= value <= 1):
            raise ValueError(
                f"{value} is a probability! Must be in range [0,1]")
        self._e = value


class MyLovelyAgent(torch.nn.Module):
    """
    This agent acts on CartPole with image as input
    Learns to approaximate the value function by using Semi-gradient TD(0)
    introduced in Reinforcement Learning An Intro. 2nd Ed. by Richard S. Sutton
    page 203
    """

    def __init__(
        self,
        image_size: int,
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
        self.transforms = transforms.Compose([
            transforms.Grayscale(),
            transforms.CenterCrop(image_size)
        ])

    def init_layers(self, action_set: set[Action]):
        """Define layers here"""
        input_width, input_height = self.input_shape, self.input_shape

        out_channels = 16
        self.conv1 = NN.Conv2d(1, out_channels, kernel_size=3)
        shape_now = (input_width-2, input_height-2, out_channels)

        self.pool1 = NN.MaxPool2d(kernel_size=2, stride=2)
        shape_now = (int(shape_now[0]/2), int(shape_now[1]/2), out_channels)

        out_channels = 32
        self.conv2 = NN.Conv2d(shape_now[2], out_channels, kernel_size=3)
        shape_now = (shape_now[0]-2, shape_now[1]-2, out_channels)

        self.pool2 = NN.MaxPool2d(kernel_size=2, stride=2)
        shape_now = (int(shape_now[0]/2), int(shape_now[1]/2), out_channels)

        out_channels = 32
        self.conv3 = NN.Conv2d(shape_now[2], out_channels, kernel_size=3)
        shape_now = (shape_now[0]-2, shape_now[1]-2, out_channels)

        self.pool3 = NN.MaxPool2d(kernel_size=2, stride=2)
        shape_now = (int(shape_now[0]/2), int(shape_now[1]/2), out_channels)

        mul = shape_now[0] * shape_now[1] * shape_now[2]

        self.flatten = NN.Flatten()
        self.fc = NN.Linear(mul, len(action_set))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Receive an image as tensor output the estimated value of each action
        TODO: Also define activations in `init_layers` function
        """
        assert issubclass(type(x), torch.Tensor)
        x = x.to(self.device)

        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        y = F.relu(self.fc(x.view(x.size(0), -1)))

        return y

    def make_up_my_mind(self, image: torch.Tensor) -> Action:
        """
        Select action using e-greedy policy on values from the
        value approaximator function a.k.a DQN. Is the naming right?
        TODO: Make this general across multiple policies
        """
        # we'll be acting on the real world, so its a test! Put it in test mode
        self.eval()
        with torch.no_grad():
            values = self(image)

        if np.random.uniform() > self.e_greedy_parameters.e:  # Select Max
            action_index = torch.argmax(values)
        else:  # Select Randomly
            action_index = torch.randint(0, len(self.action_set), (1,))

        self.train()

        return action_index

    def copy(self) -> "MyLovelyAgent":
        new_agent = MyLovelyAgent(self.input_shape, self.action_set,
                                  self.e_greedy_parameters, self.device)
        new_agent.load_state_dict(self.state_dict())
        
        return new_agent
