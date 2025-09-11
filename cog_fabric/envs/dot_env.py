import numpy as np

class EventDotEnv:
    """
    Simple event-based environment:
    - A single bright dot moves on a 2D grid with constant velocity (bounces at edges).
    - We synthesize 'events' based on intensity changes vs previous frame.
    - Observation: event histogram (2 channels: positive, negative) over the last step.
    - Target: next dot position (normalized to [0,1]^2).
    - Optional perturbation: mid-episode speed jump and brief event noise.
    """
    def __init__(self, width=64, height=64, event_threshold=0.15, max_speed=2.5, seed=42,
                 perturb=None):
        self.W = width
        self.H = height
        self.thr = event_threshold
        self.max_speed = max_speed
        self.rng = np.random.RandomState(seed)
        self.frame = np.zeros((self.H, self.W), dtype=np.float32)
        self.prev_frame = np.zeros_like(self.frame)
        self.pos = None
        self.vel = None
        self.t = 0
        # perturbation config
        self.perturb = perturb or {"enabled": False}
        self.noise_left = 0

    def reset(self):
        self.frame.fill(0.0)
        self.prev_frame.fill(0.0)
        self.pos = np.array([self.rng.uniform(5, self.W-5), self.rng.uniform(5, self.H-5)], dtype=np.float32)
        angle = self.rng.uniform(0, 2*np.pi)
        speed = self.rng.uniform(0.5, self.max_speed)
        self.vel = np.array([np.cos(angle)*speed, np.sin(angle)*speed], dtype=np.float32)
        self.t = 0
        self.noise_left = 0
        self._render_frame()
        return self._events(), self._target()

    def step(self):
        self.prev_frame[...] = self.frame
        # Move
        self.pos += self.vel
        # Bounce on edges
        if self.pos[0] < 2 or self.pos[0] > self.W-3:
            self.vel[0] *= -1
            self.pos[0] = np.clip(self.pos[0], 2, self.W-3)
        if self.pos[1] < 2 or self.pos[1] > self.H-3:
            self.vel[1] *= -1
            self.pos[1] = np.clip(self.pos[1], 2, self.H-3)
        self._render_frame()

        # Perturbation: speed jump + optional noisy window
        if self.perturb.get("enabled", False) and self.t == int(self.perturb.get("step", 200)):
            self.vel *= float(self.perturb.get("speed_multiplier", 2.0))
            self.noise_left = int(self.perturb.get("noise_steps", 0))

        self.t += 1
        return self._events(), self._target()

    def _render_frame(self):
        # Gaussian blob centered at pos
        y = np.arange(self.H)[:, None]
        x = np.arange(self.W)[None, :]
        dy = y - self.pos[1]
        dx = x - self.pos[0]
        dist2 = (dx*dx + dy*dy) / (2.0*2.0)  # sigma^2 = 4
        blob = np.exp(-dist2).astype(np.float32)
        self.frame = blob

    def _events(self):
        diff = self.frame - self.prev_frame
        pos_ev = (diff > self.thr).astype(np.float32)
        neg_ev = (diff < -self.thr).astype(np.float32)
        hist = np.stack([pos_ev, neg_ev], axis=0)

        # Optional Gaussian noise for a few steps after shock
        if self.noise_left > 0:
            std = float(self.perturb.get("noise_std", 0.0))
            if std > 0.0:
                noise = self.rng.normal(0.0, std, size=hist.shape).astype(np.float32)
                hist = np.clip(hist + noise, 0.0, 1.0)
            self.noise_left -= 1
        return hist  # (2, H, W)

    def _target(self):
        # Normalize position to [0,1]
        return (self.pos[0] / (self.W-1), self.pos[1] / (self.H-1))
