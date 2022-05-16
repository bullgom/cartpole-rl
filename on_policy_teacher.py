from agent import MyLovelyAgent
import torch
import torch.nn.functional as F
from experience import ExperienceBuffer

class MrAdamsTheTeacher:
    """Implements algorithms for the learning part"""

    def __init__(
        self, 
        agent: MyLovelyAgent, 
        discount: float, 
        optimizer: torch.optim.Optimizer, 
        device: torch.device,
        buffer: ExperienceBuffer,
        batch_size: int
    ):
        self.agent = agent
        self.discount = discount
        self.optimizer = optimizer
        self.device = device
        self.buffer = buffer
        self.bs = batch_size
    
    def loss(
        self, 
        state: torch.Tensor, 
        next_state: torch.Tensor,
        action: torch.Tensor, 
        reward: torch.Tensor, 
        done: torch.Tensor
    ):
        # tensor.max returns multiple informations. Take max values by .values
        q_scalar = self.agent(next_state).max(dim=1).values
        
        target = reward + done.logical_not() * (self.discount * q_scalar)
        values = self.agent(state)
        pred_q = values.gather(dim=1, index=action.view(1,-1)).squeeze(0)
        
        return F.mse_loss(pred_q, target)
    
    def teach_multiple(self):
        
        if len(self.buffer) < self.bs:
            return
        
        self.optimizer.zero_grad()

        last, new, r, a, done = self.buffer.sample(self.bs)

        state = last.to(self.device)
        next_state = new.to(self.device)
        action = a.to(self.device)
        reward = r.to(self.device)
        done = done.int().to(self.device)
        
        self.loss(state, next_state, action, reward, done).backward()
        self.optimizer.step()
        