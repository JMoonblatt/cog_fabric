import torch

class ESNReservoir(torch.nn.Module):
    def __init__(self, in_dim, size=512, spectral_radius=0.9, input_scale=0.5, leak_rate=0.3, sparsity=0.95, device='cpu'):
        super().__init__()
        self.N = size
        self.device = device
        # Input weights
        self.W_in = torch.randn(self.N, in_dim, device=device) * input_scale / (in_dim ** 0.5)
        # Recurrent weights (sparse)
        W = torch.randn(self.N, self.N, device=device)
        mask = (torch.rand_like(W) > sparsity).float()
        W = W * mask
        # Spectral radius normalization
        eigval = torch.linalg.eigvals(W).abs().max().real + 1e-6
        self.W = torch.nn.Parameter((W / eigval * spectral_radius), requires_grad=False)
        self.leak = leak_rate
        self.state = torch.zeros(self.N, device=device)

        # Readout to 2D position
        self.readout = torch.nn.Linear(self.N, 2, bias=True, device=device)

    def forward(self, u):
        """u: (in_dim,)"""
        with torch.no_grad():
            pre = self.W_in @ u
            rec = self.W @ self.state
            x = torch.tanh(pre + rec)
            self.state = (1 - self.leak) * self.state + self.leak * x
        # Predict next position from state
        y = self.readout(self.state)
        return y.unsqueeze(0)

    def reset_state(self):
        self.state.zero_()
