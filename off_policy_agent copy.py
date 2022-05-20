import torch
import torch.nn as NN
import torch.nn.functional as F
import numpy as np
import typing
from torchvision.transforms import transforms
from PIL import Image

Action = typing.Any



class EpsilonGreedyParameters:

    # Probability of selecting randomly
    e_start :  float
    e_end : float
    e_decrease_episode : int

    def __init__(self, e_start: float, e_end: float, episodes : int) -> None:
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
        image_size: typing.Union[int, tuple[int,int]],
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
            transforms.Resize(image_size, Image.CUBIC),
            #transforms.RandomAffine(0, translate=None, scale=[0.9, 1.1], shear=None, resample=False, fillcolor=128)
        ])

    def init_layers(self, action_set: set[Action]):
        """Define layers here"""
        if type(self.input_shape) is int:
            input_width, input_height = self.input_shape, self.input_shape
        else:
            input_width, input_height = self.input_shape

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
