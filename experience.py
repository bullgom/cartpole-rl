from torch import Tensor
import torch
from dataclasses import dataclass, astuple
import numpy as np

@dataclass
class Experience:
    last_state : Tensor
    current_state : Tensor
    reward : float
    action : int
    done : bool

    def __iter__(self):
        return iter(astuple(self))
    
class ExperienceBuffer(list):
    
    def __init__(self, max_length: int):
        assert type(max_length) is int
        self.max_length = max_length
    
    def append(self, __object: Experience) -> None:
        if not issubclass(type(__object), Experience):
            raise TypeError(f"Expected instance of an Experience, got {type(__object)} instead")
        
        if len(self) >= self.max_length:
            self.pop(0)
        return super().append(__object)
    
    def full(self) -> bool:
        return (len(self) == self.max_length)

    def sample(self, amount: int) -> tuple:
        indices = np.random.choice(len(self), amount, replace=False)
        lasts, currs, rs, actions, dones = zip(*[self[i] for i in indices])
        lasts = torch.stack(lasts)
        currs = torch.stack(currs)
        rs = torch.tensor(rs)
        actions = torch.tensor(actions)
        dones = torch.tensor(dones)
        return lasts, currs, rs, actions, dones

if __name__ == "__main__":
    eb = ExperienceBuffer(max_length=10)
    for i in range(15):
        eb.append(i)
    print(eb)
