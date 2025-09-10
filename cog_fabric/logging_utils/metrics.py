import time
import numpy as np

class MetricTracker:
    def __init__(self):
        self.records = []  # (t, err, mode)

    def log(self, step_idx, err, mode):
        self.records.append((step_idx, float(err), mode))

    def adaptation_half_life(self):
        # crude: find how many steps it takes to reduce error by 50% after a spike
        errs = [r[1] for r in self.records]
        if len(errs) < 10:
            return None
        peak_idx = max(range(len(errs)), key=lambda i: errs[i])
        peak = errs[peak_idx]
        target = peak * 0.5
        for j in range(peak_idx+1, len(errs)):
            if errs[j] <= target:
                return j - peak_idx
        return None

    def mean_err(self):
        if not self.records:
            return None
        return float(np.mean([r[1] for r in self.records]))
