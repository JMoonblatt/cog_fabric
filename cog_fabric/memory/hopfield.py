import torch

class TinyHopfield:
    """Very small associative memory that stores vector patterns and retrieves by nearest prototype."""
    def __init__(self, dim, max_patterns=64, device='cpu'):
        self.dim = dim
        self.max_patterns = max_patterns
        self.device = device
        self.patterns = []  # list of (tensor, tag)

    def store(self, v, tag=None):
        v = v.detach().to(self.device).float()
        v = v / (v.norm() + 1e-6)
        if len(self.patterns) >= self.max_patterns:
            self.patterns.pop(0)
        self.patterns.append((v, tag))

    def retrieve(self, q, topk=1):
        if not self.patterns:
            return None
        q = q.detach().to(self.device).float()
        q = q / (q.norm() + 1e-6)
        sims = torch.tensor([torch.dot(q, p) for (p, _) in self.patterns], device=self.device)
        idx = torch.topk(sims, k=min(topk, len(self.patterns))).indices.tolist()
        return [self.patterns[i] for i in idx]
