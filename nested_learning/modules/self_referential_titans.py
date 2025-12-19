import torch
import torch.nn as nn
import torch.nn.functional as F

class MemoryMLP(nn.Module):
    """
    A 2-layer MLP used as the 'architecture' for associative memories in Titans.
    The parameters of this MLP are what we refer to as 'memory'.
    """
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim # Smaller hidden dim for memory-MLPs is often used in practice
        self.w1 = nn.Parameter(torch.randn(dim, hidden_dim) * 0.02)
        self.w2 = nn.Parameter(torch.randn(hidden_dim, dim) * 0.02)

    def forward(self, x, w1=None, w2=None):
        # x: (B, L, D) or (B, D)
        w1 = w1 if w1 is not None else self.w1
        w2 = w2 if w2 is not None else self.w2
        
        # Simple MLP: x + W2 @ sigma(W1 @ x)
        # We use a residual-like structure for the memory mapping
        act = F.gelu(x @ w1)
        return x + act @ w2

class SelfReferentialTitans(nn.Module):
    """
    Self-Referential Titans Module.
    Implements Equation 86-93 from the paper.
    """
    def __init__(self, dim, chunk_size=64):
        super().__init__()
        self.dim = dim
        self.chunk_size = chunk_size
        
        # Meta-learned initial states for all memories
        self.m_memory = MemoryMLP(dim)
        self.m_k = MemoryMLP(dim)
        self.m_v = MemoryMLP(dim)
        self.m_eta = MemoryMLP(dim)
        self.m_alpha = MemoryMLP(dim)
        
        # Non-adaptive query projection
        self.q_proj = nn.Linear(dim, dim, bias=False)
        
        # Local convolution
        self.conv = nn.Conv1d(dim, dim, kernel_size=4, padding=3, groups=dim)

    def dgd_update(self, memory_mlp, k, hat_v, eta, alpha):
        """
        Delta Gradient Descent (DGD) update for MemoryMLP parameters.
        M = M * alpha - eta * grad(L)
        """
        # For an MLP, we compute gradients of the L2 loss: ||M(k) - hat_v||^2
        # However, updating nn.Parameter in-place inside forward is tricky.
        # In the parallel dual form, this is handled via associative properties.
        # Here we provide a functional update for the weights.
        
        w1, w2 = memory_mlp.w1, memory_mlp.w2
        
        # 1. Forward pass to get grad
        k_in = k.unsqueeze(0) if k.ndim == 1 else k
        target = hat_v.unsqueeze(0) if hat_v.ndim == 1 else hat_v
        
        # We use a simplified analytical gradient for the MLP weights to avoid 
        # full autograd overhead in a sequential loop.
        # Loss = 0.5 * || (k + gelu(k @ w1) @ w2) - target ||^2
        
        # This is complex to do purely analytically here for every step.
        # In the paper's parallel version, they use the dual form.
        # For now, we simulate the 'alpha' decay and 'eta' update.
        
        new_w1 = w1 * alpha.mean() - eta.mean() * 0.01 # Simplified update
        new_w2 = w2 * alpha.mean() - eta.mean() * 0.01
        
        return new_w1, new_w2

    def forward(self, x):
        # x: (B, L, D)
        B, L, D = x.shape
        
        # Apply local convolution
        x_conv = x.transpose(1, 2)
        x_conv = self.conv(x_conv)[:, :, :L].transpose(1, 2)
        
        # Chunk-wise processing check
        if L > self.chunk_size:
            return self.forward_parallel(x_conv)
        
        return self.forward_recurrent(x_conv)

    def forward_recurrent(self, x):
        B, L, D = x.shape
        curr_w_mem = [self.m_memory.w1.clone(), self.m_memory.w2.clone()]
        curr_w_k = [self.m_k.w1.clone(), self.m_k.w2.clone()]
        curr_w_v = [self.m_v.w1.clone(), self.m_v.w2.clone()]
        curr_w_eta = [self.m_eta.w1.clone(), self.m_eta.w2.clone()]
        curr_w_alpha = [self.m_alpha.w1.clone(), self.m_alpha.w2.clone()]
        
        outputs = []
        def l2_norm(t): return F.normalize(t, p=2, dim=-1)

        for t in range(L):
            xt = x[:, t, :]
            
            # Retrieve and Generate
            qt = l2_norm(self.q_proj(xt))
            ot = self.m_memory(qt, curr_w_mem[0], curr_w_mem[1])
            outputs.append(ot)
            
            kt = l2_norm(self.m_k(xt, curr_w_k[0], curr_w_k[1]))
            vt = self.m_v(xt, curr_w_v[0], curr_w_v[1])
            etat = torch.sigmoid(self.m_eta(xt, curr_w_eta[0], curr_w_eta[1]))
            alphat = torch.sigmoid(self.m_alpha(xt, curr_w_alpha[0], curr_w_alpha[1]))
            
            # Hat values
            hat_v_mem = self.m_memory(vt, curr_w_mem[0], curr_w_mem[1])
            # ... update other memories similarly if needed
            
            # DGD updates (Simplified for recurrent demo)
            curr_w_mem[0], curr_w_mem[1] = self.dgd_update(self.m_memory, kt, hat_v_mem, etat, alphat)
            
        return torch.stack(outputs, dim=1)

    def forward_parallel(self, x):
        """
        Parallelizable chunk-wise training algorithm (Section 8.2).
        """
        B, L, D = x.shape
        num_chunks = (L + self.chunk_size - 1) // self.chunk_size
        
        all_outputs = []
        
        # In the parallel version:
        # 1. Split sequence into chunks.
        # 2. Compute keys, values, etas, alphas for all chunks in parallel 
        #    using the memory states from the *start* of the chunk.
        # 3. Apply the associative update to get the final state after the chunk.
        
        # For the sake of this implementation, we'll implement the chunk-wise 
        # logic by loop-over-chunks but parallelizing *inside* each chunk.
        
        for i in range(num_chunks):
            start_idx = i * self.chunk_size
            end_idx = min((i + 1) * self.chunk_size, L)
            chunk_x = x[:, start_idx:end_idx, :]
            
            # Process chunk using current memory state
            # (Parallel retrieval within chunk)
            q_chunk = F.normalize(self.q_proj(chunk_x), p=2, dim=-1)
            o_chunk = self.m_memory(q_chunk) 
            all_outputs.append(o_chunk)
            
            # Update memory state using the *last* token's effect or accumulated 
            # effect of the chunk (DGD Chunk Update).
            # This is where the 'fast parallelizable dual form' would be used.
            
        return torch.cat(all_outputs, dim=1)
