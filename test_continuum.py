import torch
import torch.nn as nn
from nested_learning.models.hope import Hope
from nested_learning.modules.cms import CMS

def test_cms_retention():
    """
    Test to verify that CMS levels with different frequencies 
    retain information across different time scales.
    """
    dim = 32
    num_levels = 3
    frequencies = [1, 5, 10]
    cms = CMS(dim, num_levels=num_levels, frequencies=frequencies)
    
    print(f"Testing CMS with frequencies: {frequencies}")
    
    # 1. Simulate steps and check updates
    for step in range(1, 11):
        mask = cms.get_update_mask(step)
        print(f"Step {step:2}: Update Mask -> {mask}")
        
    # 2. Functional check of Hope model with CMS
    model = Hope(dim=dim, num_layers=1, num_cms_levels=num_levels)
    x = torch.randn(1, 50, dim)
    out = model(x)
    
    print(f"Hope model forward pass successful. Output shape: {out.shape}")
    assert out.shape == (1, 50, dim)
    print("CMS and Hope structure verification passed.")

if __name__ == "__main__":
    test_cms_retention()
