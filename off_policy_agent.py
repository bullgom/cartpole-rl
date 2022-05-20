import torch
import torch.nn as nn
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
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)


        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_width)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_height)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Receive an image as tensor output the estimated value of each action
        TODO: Also define activations in `init_layers` function
        """
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
