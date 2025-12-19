import torch
import torch.nn as nn
from nested_learning.models.hope import Hope
from nested_learning.optimizers.m3 import M3
import time

def train_demo():
    # 1. Setup Hyperparameters
    dim = 64
    seq_len = 128
    batch_size = 4
    num_layers = 2
    num_cms_levels = 3
    lr = 1e-4
    
    # 2. Initialize Model
    print(f"Initializing Hope model (dim={dim}, layers={num_layers}, CMS levels={num_cms_levels})...")
    model = Hope(dim=dim, num_layers=num_layers, num_cms_levels=num_cms_levels)
    
    # 3. Initialize Optimizer (M3)
    # Using small frequency for the demo
    optimizer = M3(model.parameters(), lr=lr, frequency=5, alpha=0.1)
    
    # 4. Generate Synthetic Data
    # Task: Sequence-to-sequence (e.g., identity or shift)
    input_data = torch.randn(batch_size, seq_len, dim)
    target_data = torch.roll(input_data, shifts=1, dims=1) # Shift task
    
    criterion = nn.MSELoss()
    
    # 5. Training Loop
    print("Starting training loop...")
    start_time = time.time()
    for epoch in range(10):
        optimizer.zero_grad()
        
        # Forward pass
        output = model(input_data)
        
        # Loss computation
        loss = criterion(output, target_data)
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        
        if (epoch + 1) % 2 == 0:
            print(f"Epoch [{epoch+1}/10], Loss: {loss.item():.4f}")
            
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    train_demo()
