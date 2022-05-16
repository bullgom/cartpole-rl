from agent import MyLovelyAgent
import torch
import torch.nn.functional as F
from experience import ExperienceBuffer

class MrAdamsTheTeacher:
    """Implements algorithms for the learning part"""

    def __init__(
        self, 
        target_agent: MyLovelyAgent,
        policy_agent: MyLovelyAgent, 
        discount: float, 
        optimizer: torch.optim.Optimizer, 
        device: torch.device,
        buffer: ExperienceBuffer,
        batch_size: int
    ):
        self.target = target_agent
        self.target.eval()
        self.policy = policy_agent
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
        q_scalar = self.target(next_state).max(dim=1).values
        not_done = done.logical_not()
        bootstrap = not_done * (self.discount * q_scalar)
        target = (reward + bootstrap).unsqueeze(0)
        
        values = self.policy(state)
        action_batch = action.view(1, -1)
        pred_q = values.gather(dim=1, index=action_batch)
        
        return F.huber_loss(pred_q, target)
    
    def teach_multiple(self) -> torch.Tensor:
        
        if len(self.buffer) < self.bs:
            return
        
        self.optimizer.zero_grad()

        last, new, r, a, done = self.buffer.sample(self.bs)

        state = last.to(self.device)
        next_state = new.to(self.device)
        action = a.to(self.device)
        reward = r.to(self.device)
        done = done.int().to(self.device)
        
        loss_tensor = self.loss(state, next_state, action, reward, done)
        loss_tensor.backward()
        self.optimizer.step()
        
        return loss_tensor

