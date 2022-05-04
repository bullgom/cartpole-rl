from agent import MyLovelyAgent
import torch
import torch.nn.functional as F

class MrAdamsTheTeacher:
    """Implements algorithms for the learning part"""

    def __init__(
        self, 
        agent: MyLovelyAgent, 
        discount: float, 
        optimizer: torch.optim.Optimizer, 
        device: torch.device
    ):
        self.agent = agent
        self.discount = discount
        self.optimizer = optimizer
        self.device = device

    def target_value(self, reward: float, new_state: torch.Tensor) -> torch.Tensor:
        """
        Using the bellman eq. the true value is estimated as R + v(S_(t+1))
        TODO: Find the proof for why this works? I only know the equation
        """

        return reward + self.discount * self.agent(new_state)

    def estimate_value(self, last_state: torch.Tensor) -> torch.Tensor:
        """
        Value Estimate of the current state
        TODO: Reuse the results of `make_up_my_mind`, OR use replay buffer
        """
        return self.agent(last_state)

    def loss(self, reward: float, new_state: torch.Tensor, last_state: torch.Tensor) -> torch.Tensor:
        """Compute loss"""
        target = self.target_value(reward, new_state)
        prediction = self.estimate_value(last_state)
        return F.mse_loss(prediction, target)

    def teach_a_problem(self, reward: float, new_state: torch.Tensor, last_state: torch.Tensor):
        """Backpropagate for given state"""
        self.optimizer.zero_grad()
        
        self.loss(reward, new_state.to(self.device), last_state.to(self.device)).backward()
        
        self.optimizer.step()
        
