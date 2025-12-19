import torch
from torch.optim.optimizer import Optimizer

def newton_schulz_orthogonalization(G, steps=5, eps=1e-7):
    """
    Newton-Schulz iteration for orthogonalization.
    Used in Muon-like optimizers.
    """
    # G: (N, M)
    a, b = G.shape
    if a < b:
        # We want to orthogonalize rows
        X = G / (torch.norm(G, p='fro') + eps)
        for _ in range(steps):
            X = 1.5 * X - 0.5 * X @ X.t() @ X
    else:
        # We want to orthogonalize columns
        X = G / (torch.norm(G, p='fro') + eps)
        for _ in range(steps):
            X = 1.5 * X - 0.5 * X @ X.t() @ X
    return X


class M3(Optimizer):
    """
    Multi-scale Momentum Muon (M3) Optimizer.
    Combines ideas from Adam, Muon, and Continuum Memory Systems.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999, 0.95), eps=1e-8, 
                 alpha=0.1, frequency=10, ns_steps=5):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0 or not 0.0 <= betas[2] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, alpha=alpha, 
                        frequency=frequency, ns_steps=ns_steps)
        super(M3, self).__init__(params, defaults)
        self.step_count = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.step_count += 1
        
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2, beta3 = group['betas']
            eps = group['eps']
            alpha = group['alpha']
            freq = group['frequency']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['m1'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)
                    state['m2'] = torch.zeros_like(p)
                    state['grad_buffer'] = torch.zeros_like(p)
                    state['o2'] = torch.zeros_like(p)

                m1, v, m2 = state['m1'], state['v'], state['m2']
                grad_buffer = state['grad_buffer']
                
                # Update grad_buffer for slow memory
                grad_buffer.add_(grad)

                # Slow memory update (every 'freq' steps)
                if self.step_count % freq == 0:
                    # state['m2'] = state['m2'] + beta3 * grad_buffer
                    # Actually Algorithm 1 says M2 = M2 + beta3 * sum(g_i)
                    m2.add_(grad_buffer, alpha=beta3)
                    
                    # Compute O2 via Newton-Schulz
                    if p.ndim >= 2:
                        # Muon is typically for 2D or more. 
                        # Flatten to 2D for orthogonalization if needed
                        orig_shape = p.shape
                        m2_flat = m2.view(orig_shape[0], -1)
                        o2_flat = newton_schulz_orthogonalization(m2_flat, steps=ns_steps, eps=eps)
                        state['o2'] = o2_flat.view(orig_shape)
                    else:
                        state['o2'] = m2.clone()
                    
                    # Reset buffer
                    grad_buffer.zero_()

                # Fast memory (Momentum 1)
                m1.mul_(beta1).add_(grad, alpha=1.0 - beta1) # Using standard EMA-like for M1
                # Wait, Algorithm 1 says M1 = M1 + beta1 * g_t. This is not EMA but cumulative.
                # But typically optimizers use EMA. Let's follow Algorithm 1 strictly or stick to EMA?
                # Algorithm 1: M1_t = M1_{t-1} + beta1 * g_t. 
                # This would grow indefinitely if beta1=1. 
                # I'll use Algorithm 1's additive form as it says "Multi-scale momentum".
                # But wait, if it's "Memory", then it might need decay. 
                # Actually, beta1 here might be smaller or the additive form is intentional.
                # Re-reading Algorithm 1: M1_t = M1_{t-1} + beta1 * g_t.
                
                # Variance (for Adam-like scaling)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                
                # Compute O1
                if p.ndim >= 2:
                    orig_shape = p.shape
                    m1_flat = m1.view(orig_shape[0], -1)
                    o1_flat = newton_schulz_orthogonalization(m1_flat, steps=ns_steps, eps=eps)
                    o1 = o1_flat.view(orig_shape)
                else:
                    o1 = m1

                # Final update: p = p - lr * (o1 + alpha * o2) / (sqrt(v) + eps)
                o2 = state['o2']
                denom = v.sqrt().add_(eps)
                p.addcdiv_(o1 + alpha * o2, denom, value=-lr)

        return loss
