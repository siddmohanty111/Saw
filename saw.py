"""
A file containing the programmatic definition of Saw.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

r = random.Random()

class LearnableActivation(nn.Module):
    """
    Base class for learnable activation functions. Subclasses should implement the actual activation logic in the forward method.
    """
    def __init__(self, learnable=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.learnable = learnable

class Saw(LearnableActivation):
    """
    Activation function with slope 1 on the extreme ends and slope -1 in the middle. Function is continuous.
    """

    def __init__(self, left_break : float = -1.0, right_break : float = 1.0, learnable = False):
        super().__init__(learnable=learnable)
        
        if learnable:
            self.left_break = nn.Parameter(torch.tensor(float(left_break)))
            self.right_break = nn.Parameter(torch.tensor(float(right_break)))
        else:
            self.register_buffer('left_break', torch.tensor(float(left_break)))
            self.register_buffer('right_break', torch.tensor(float(right_break)))

    def forward(self, x):
        # Vectorized implementation using torch.where for efficiency
        # Region 1: x < left_break -> out = x - left_break
        # Region 2: left_break <= x < right_break -> -x
        # Region 3: x >= right_break -> out = x - right_break
        
        result = torch.where(x < self.left_break,
                            x - 2*self.left_break,
                            torch.where(x < self.right_break,
                                        -x,
                                        x - 2*self.right_break))
        return result
