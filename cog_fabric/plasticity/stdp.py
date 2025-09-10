import torch

class HebbSTDP(torch.nn.Module):
    """A simple local plasticity layer: y = W x, with Hebbian/STDP-like updates when gated."""
    def __init__(self, in_dim, out_dim, lr=5e-4, decay=1e-4, device='cpu'):
        super().__init__()
        self.W = torch.nn.Parameter(0.01 * torch.randn(out_dim, in_dim, device=device))
        self.lr = lr
        self.decay = decay
        self.device = device

    def forward(self, x):
        # x: (B, in_dim)
        return x @ self.W.T  # (B, out_dim)

    @torch.no_grad()
    def plastic_update(self, pre, post, gate=True):
        """pre: (B, in_dim), post: (B, out_dim)"""
        if not gate:
            return
        # Simple 3-factor-like rule: dW = lr * (post^T * pre) - decay * W
        # Here we use batch outer product averaged over batch.
        B = pre.shape[0]
        dW = (post.T @ pre) / max(1, B)
        self.W += self.lr * dW - self.decay * self.W
