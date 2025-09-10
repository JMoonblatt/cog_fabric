import numpy as np
import torch

from cog_fabric.plasticity.stdp import HebbSTDP
from cog_fabric.reservoir.reservoir import ESNReservoir
from cog_fabric.memory.hopfield import TinyHopfield
from cog_fabric.router.router import SimpleRouter

def flatten_events(ev):
    # ev shape: (2, H, W) -> vector
    return torch.from_numpy(ev.reshape(1, -1).astype(np.float32))

class Agent:
    def __init__(self, env_shape, cfg, device='cpu'):
        C, H, W = env_shape
        self.device = device
        self.in_dim = C*H*W
        self.cfg = cfg

        # Plasticity layer maps event histogram to reservoir input space
        self.plastic = HebbSTDP(self.in_dim, cfg['reservoir']['size'],
                                lr=cfg['plasticity']['lr'],
                                decay=cfg['plasticity']['decay'],
                                device=device)

        # Reservoir with readout to 2D position
        self.res = ESNReservoir(self.in_dim, size=cfg['reservoir']['size'],
                                spectral_radius=cfg['reservoir']['spectral_radius'],
                                input_scale=cfg['reservoir']['input_scale'],
                                leak_rate=cfg['reservoir']['leak_rate'],
                                sparsity=cfg['reservoir']['sparsity'],
                                device=device)

        self.optim = torch.optim.Adam(self.res.readout.parameters(), lr=cfg['reservoir']['lr'])

        # Tiny memory
        self.mem = TinyHopfield(dim=self.in_dim, max_patterns=cfg['memory']['max_patterns'], device=device)

        # Router
        self.router = SimpleRouter(window=cfg['router']['error_window'],
                                   novelty_beta=cfg['router']['novelty_beta'],
                                   error_threshold=cfg['router']['error_threshold'])

        # Replay buffer of (ev_vec, target)
        self.replay = []

    def reset(self):
        self.res.reset_state()

    def step(self, ev, target, gates, features):
        plasticity_gate, routing_gate, theta_phase, gamma_phase = gates
        ev_vec = flatten_events(ev).to(self.device) # (1, in_dim)
        target_t = torch.tensor(target, dtype=torch.float32, device=self.device).view(1, 2)

        # Optionally retrieve memory pattern when error is high / router says so
        use_memory = False
        if features['router_enabled'] and routing_gate:
            decision = self.router.decide()
            use_memory = decision['use_memory']
        else:
            decision = {'use_memory': False, 'allow_plasticity': True}

        ev_in = ev_vec
        if features['memory_enabled'] and use_memory and self.mem.patterns:
            # Blend with nearest memory prototype to fill in missing data
            top = self.mem.retrieve(ev_vec.view(-1))
            if top:
                proto, _ = top[0]
                proto = proto.view(1, -1)
                ev_in = 0.7 * ev_vec + 0.3 * proto  # simple completion

        # Pass through plasticity layer to reservoir input space (linear map)
        # For simplicity we feed directly to reservoir's input 'u' by reusing ev_vec.
        # The plastic layer can be seen as a learned preprocessor (we still compute its output for learning signal).
        y_pred = self.res(ev_in.view(-1))
        loss = torch.nn.functional.mse_loss(y_pred, target_t)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # Local plasticity update (Hebbian-style), only when gated & enabled
        if features['plasticity_enabled'] and decision.get('allow_plasticity', True) and plasticity_gate:
            # Use reservoir state as 'post', ev_vec as 'pre'
            with torch.no_grad():
                post = self.res.state.view(1, -1)
                pre = ev_vec
                self.plastic.plastic_update(pre, post, gate=True)

        # Update router with current error
        self.router.update_error(float(loss.item()))

        # Store to memory occasionally (when error is high or novelty threshold exceeded)
        if features['memory_enabled'] and float(loss.item()) > (self.router.threshold * 0.8):
            self.mem.store(ev_vec.view(-1), tag=None)

        # Add to replay buffer
        self.replay.append((ev_vec.cpu(), target_t.cpu()))
        if len(self.replay) > 512:
            self.replay.pop(0)

        return float(loss.item())

    def sleep_replay(self, epochs=2):
        if not self.replay:
            return 0.0
        tot = 0.0
        for _ in range(epochs):
            for ev_vec, target_t in self.replay:
                ev_vec = ev_vec.to(self.device)
                target_t = target_t.to(self.device)
                y_pred = self.res(ev_vec.view(-1))
                loss = torch.nn.functional.mse_loss(y_pred, target_t)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                tot += float(loss.item())
        return tot / max(1, epochs * len(self.replay))
