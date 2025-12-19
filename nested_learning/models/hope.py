import torch
import torch.nn as nn
from nested_learning.modules.self_referential_titans import SelfReferentialTitans
from nested_learning.modules.cms import CMS

class Hope(nn.Module):
    """
    Hope Architecture.
    Assembles Self-Referential Titans followed by a Continuum Memory System (CMS).
    """
    def __init__(self, dim, num_layers=2, num_cms_levels=3, chunk_size=64):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                'titans': SelfReferentialTitans(dim, chunk_size=chunk_size),
                'cms': CMS(dim, num_levels=num_cms_levels),
                'norm1': nn.LayerNorm(dim),
                'norm2': nn.LayerNorm(dim)
            })
            self.layers.append(layer)
            
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (B, L, D)
        for layer in self.layers:
            # 1. Self-Referential Titans
            resid = x
            x = layer['norm1'](x)
            x = layer['titans'](x)
            x = x + resid
            
            # 2. CMS (Continuum Memory System)
            resid = x
            x = layer['norm2'](x)
            x = layer['cms'](x)
            x = x + resid
            
        return self.final_norm(x)
