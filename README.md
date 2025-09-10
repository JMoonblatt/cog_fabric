## User Guide

Cog Fabric is an experimental framework for studying adaptive and emergent intelligence. It combines an event-based environment with modules for rhythms, plasticity, memory, and routing. The goal is to explore how systems respond to shocks and adapt over time.

### Getting Started

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

Cog Fabric is not about “winning” benchmarks but about **studying adaptation** — how quickly and robustly artificial systems recover when the world changes.

```

---

Do you want me to also prep a **second short “Quickstart” snippet** (just 3–4 lines with install, run baseline, open UI) that you can put at the *top* of the README so newcomers see it right away?
```
