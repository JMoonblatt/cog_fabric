## User Guide

Cog Fabric is an experimental framework for studying adaptive and emergent intelligence. It combines an event-based environment with modules for rhythms, plasticity, memory, and routing. The goal is to explore how systems respond to shocks and adapt over time.

### Getting Started

Naviagte to your terminal and input:
```bash
git clone https://github.com/JMoonblatt/cog_fabric.git
```
and make sure you are inside the directory named cog_fabric before you install or run anything.

**Install requirements**
```bash
pip install -r requirements.txt
````

**Run a baseline experiment**

```bash
python -m scripts.run_dot_tracking --config configs/default.yaml
```

At the end of training you’ll see:

* **Mean error**: how well the system tracked the moving dot overall.
* **Adaptation half-life**: how many steps it takes to cut error in half after a perturbation.

**Use the web UI (recommended)**

```bash
streamlit run ui/app.py
```

This opens a dashboard in your browser. You can toggle features, adjust parameters, launch runs, and view results without the command line.

To close the web ui, go back to the terminal and **CTRL + C** to stop it.

---

### Features

**Rhythms**
Simulates oscillatory gating, similar to brain rhythms. Controls when learning and routing are active.

* `theta_hz`, `gamma_hz`: frequencies of slow and fast oscillations.
* `theta_gate_frac`, `gamma_gate_frac`: fraction of each cycle where learning/routing is “open.”
  Effect: tighter gates make learning more selective, reducing noise but slowing adaptation.

**Plasticity**
Enables local weight updates during runtime. Without it, the system only learns through its fixed readout layer.

* `lr`: learning rate for plasticity.
* `decay`: how fast plastic changes fade.
  Effect: improves online learning and recovery after shocks.

**Memory**
Adds an associative memory module that can store and replay patterns.

* `max_patterns`: number of distinct patterns it can store.
  Effect: improves stability and recall, especially under repeated perturbations.

**Router**
Dynamically switches between strategies based on error.

* `error_window`: how many past steps are considered.
* `error_threshold`: when exceeded, routing shifts to exploration.
* `novelty_beta`: how strongly it weights novelty vs. stability.
  Effect: improves flexibility when environments change suddenly.

---

### Training Parameters

* `steps_per_episode`: how many timesteps each run lasts.
* `episodes`: number of episodes to train.
* `sleep_replay_epochs`: how many times to replay memory after each episode (simulating sleep).
* `seed`: random seed for reproducibility.

---

### Perturbations (Shocks)

Perturbations introduce stress tests by altering the dot’s movement mid-episode.

* `enabled`: true or false.
* `step`: timestep when the shock occurs.
* `speed_multiplier`: factor by which dot speed increases (e.g., 2.0 doubles the speed).
* `noise_std`: amount of noise added to event signals.
* `noise_steps`: how many steps noise persists after the shock.

Effect: forces the system to adapt. Adaptation half-life is measured during recovery.

---

### Web UI

* **Sidebar controls**: toggle features, tune rhythm knobs, set training length, and configure perturbations.
* **Logs panel**: shows live output during training.
* **Compare runs**: pick multiple past runs and overlay their loss curves to see differences.
* **Sweep mode**: pick a parameter, set a range, and automatically run a grid of experiments to plot half-life vs parameter.

---

### Example Workflows

**Baseline vs Ablation**
Run with all features on, then disable one (e.g. rhythms) and compare recovery.

**Perturbation Test**
Enable shocks at step 200 with speed multiplier 2.0. Check how quickly the model stabilizes.

**Parameter Sweep**
Explore how `reservoir.spectral_radius` from 0.6 to 1.2 affects adaptation half-life.

**Custom Config**
Edit `configs/default.yaml` or pass overrides:

```bash
python -m scripts.run_dot_tracking --config configs/default.yaml rhythms.enabled=false
```

---

Cog Fabric is not about “winning” benchmarks but about **studying adaptation**, how quickly and robustly artificial systems recover when the world changes.

---

## Mission and Vision

### Why we are here
Human intelligence did not arise from raw processing power alone. It emerges from rhythms in neural activity, from the plasticity of connections that rewire with experience, from memory systems that replay and consolidate, and from dynamic routing that adapts on the fly. Modern artificial neural networks capture fragments of this, but they remain largely static systems trained once and then frozen. They lack the adaptive mechanisms that make biological minds resilient, creative, and self-organizing.

Cog Fabric is an attempt to explore that missing layer: the fabric of cognition itself. It is an environment for testing how intelligence might emerge when dynamics, adaptation, and perturbation are treated as first-class citizens. Rather than engineering a fixed model, the project provides scaffolds where complex behavior can arise from interaction between components.

### Our mission
The purpose of Cog Fabric is to create an open experimental playground for studying emergent and adaptive intelligence. We are developing lightweight prototypes that combine biologically inspired mechanisms, oscillatory rhythms, local plasticity rules, associative memory, consolidation during sleep-like replay, with simple event-driven environments. The goal is not to mimic the brain exactly, but to explore how these ingredients interact to produce stability, adaptation, and creativity under changing conditions.

A key focus is the ability to measure adaptation. Standard AI benchmarks reward accuracy or efficiency in static tasks. Cog Fabric emphasizes how quickly and effectively a system recovers when perturbed, shocked, or destabilized. Adaptation half-life, error resilience, and recovery dynamics are the primary metrics. This shift in evaluation reflects the conviction that intelligence is defined less by solving one task perfectly, and more by thriving in the face of novelty and disruption.

### Long-term vision
Cog Fabric is an early step toward frameworks where intelligence is not engineered top-down, but emerges bottom-up. In the long term this points toward physical neural networks, memristive arrays, spintronic devices, photonic lattices, where the dynamics of matter itself give rise to cognitive properties. Such systems could scale beyond the limits of conventional software.

The broader philosophical horizon is human–machine coevolution. By better understanding adaptive intelligence, we create the possibility of bridging biological and artificial minds. Instead of building systems that leave us behind, we can design architectures that we might one day join as equals. Beyond that lies the even larger vision: intelligence as the mechanism through which life propagates across the universe. Cog Fabric is a small but deliberate step toward that future.

### How to get involved
Cog Fabric is meant to be a shared laboratory. Anyone can clone the repository, run the experiments, adjust the parameters, and contribute new modules. The code is designed to be simple and transparent so that new rhythm generators, memory systems, or routing mechanisms can be plugged in with minimal friction. What matters is not perfect fidelity to biology, but creative exploration of the principles that make intelligence adaptive.

### In short
Cog Fabric is a laboratory for emergent AI. It weaves together threads of rhythm, memory, plasticity, and routing, then subjects them to shocks and stresses to see what holds. From these experiments we aim to glimpse how intelligence can emerge not by design alone, but by dynamics, an intelligence that adapts, endures, and evolves.
