import numpy as np

class RhythmGater:
    """Generates theta/gamma phases and exposes gates for plasticity and routing."""
    def __init__(self, theta_hz=6.0, gamma_hz=40.0, theta_gate_frac=0.25, gamma_gate_frac=0.5, dt=0.01):
        self.theta_hz = theta_hz
        self.gamma_hz = gamma_hz
        self.theta_gate_frac = theta_gate_frac
        self.gamma_gate_frac = gamma_gate_frac
        self.dt = dt
        self.t = 0.0

    def step(self):
        self.t += self.dt
        theta_phase = (self.t * self.theta_hz) % 1.0
        gamma_phase = (self.t * self.gamma_hz) % 1.0
        plasticity_gate = theta_phase < self.theta_gate_frac
        routing_gate = gamma_phase < self.gamma_gate_frac
        return plasticity_gate, routing_gate, theta_phase, gamma_phase
