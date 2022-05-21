from agent import MyLovelyAgent
import torch
import torch.nn.functional as F
from er import ReplayMemory, Transition


class MrAdamsTheTeacher:
    """Implements algorithms for the learning part"""

    def __init__(
        self,
        target_agent: MyLovelyAgent,
        policy_agent: MyLovelyAgent,
        discount: float,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        buffer: ReplayMemory,
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
        q_scalar = self.target(next_state).max(dim=1)[0].detach()
        not_done = done.logical_not()
        bootstrap = not_done * (self.discount * q_scalar)
        target = (reward + bootstrap).unsqueeze(1)

        values = self.policy(state)
        action_batch = action.view(self.bs, -1)
        pred_q = values.gather(dim=1, index=action_batch)

        return F.smooth_l1_loss(pred_q, target)

    def teach_multiple(self) -> torch.Tensor:
        if len(self.buffer) < self.bs:
            return
        transitions = self.buffer.sample(self.bs)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.    next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        output = self.policy(state_batch)
        state_action_values = output.gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.bs, device=self.device)
        next_state_values[non_final_mask] = self.target(
            non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values

        expected_state_action_values = (
            next_state_values * self.discount) + reward_batch

        # Compute Huber loss
        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss
