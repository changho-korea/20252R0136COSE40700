import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPBlock(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()

    def forward(self, x):
        # x: (B, L, D) or (B, D)
        return self.fc2(self.act(self.fc1(x)))

class CMS(nn.Module):
    """
    Continuum Memory System (CMS).
    A chain of MLP blocks with different update frequencies.
    """
    def __init__(self, dim, num_levels=3, frequencies=None, hidden_dim=None):
        super().__init__()
        self.dim = dim
        self.num_levels = num_levels
        
        if frequencies is None:
            # Example frequencies: 1, 10, 100
            frequencies = [10 ** i for i in range(num_levels)]
        self.frequencies = frequencies
        
        self.blocks = nn.ModuleList([
            MLPBlock(dim, hidden_dim) for _ in range(num_levels)
        ])
        
        # In a real "Nested Learning" implementation, these blocks would have 
        # fast weights updated in-context. For the sake of this implementation, 
        # we will provide a structure that can be used for both standard training 
        # and hypothetical in-context adaptation.
        
    def forward(self, x):
        """
        Forward pass through the chain of MLP blocks.
        y = MLP_k(MLP_{k-1}(...MLP_1(x)))
        """
        # x: (B, L, D)
        out = x
        for block in self.blocks:
            # Residual connection is common in these types of architectures
            out = out + block(out)
        return out

    def get_update_mask(self, step):
        """
        Returns which levels should be updated at this step.
        """
        mask = [step % f == 0 for f in self.frequencies]
        return mask
