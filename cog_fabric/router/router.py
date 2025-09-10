import numpy as np

class SimpleRouter:
    """Routes module usage based on rolling prediction error and novelty."""
    def __init__(self, window=20, novelty_beta=0.7, error_threshold=0.07):
        self.window = window
        self.beta = novelty_beta
        self.err_hist = []
        self.err_mean = 1.0
        self.err_var = 1.0
        self.threshold = error_threshold

    def update_error(self, err):
        self.err_hist.append(err)
        if len(self.err_hist) > self.window:
            self.err_hist.pop(0)
        # Exponential moving stats
        self.err_mean = self.beta * self.err_mean + (1 - self.beta) * err
        self.err_var = self.beta * self.err_var + (1 - self.beta) * (err - self.err_mean) ** 2

    def decide(self):
        """Return dict of which modules to engage."""
        engage_memory = (self.err_mean > self.threshold) or (len(self.err_hist) > 5 and np.mean(self.err_hist[-5:]) > self.threshold)
        engage_plasticity = self.err_mean > 0.02  # allow some learning when error isn't tiny
        return {
            "use_memory": bool(engage_memory),
            "allow_plasticity": bool(engage_plasticity),
        }
