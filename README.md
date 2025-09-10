# cog_fabric: A tiny lab for emergent, adaptive AI

This repo is a **minimal, runnable scaffold** for experimenting with:
- Event-based perception
- Rhythmic (theta/gamma) gating
- Local plasticity (Hebbian / STDP-like)
- A simple reservoir (echo-state network)
- A toy associative memory (Hopfield placeholder)
- A dynamic router that routes modules based on prediction error / novelty
- Sleep/replay consolidation

No special hardware required: Python 3.10+, NumPy, PyTorch, PyYAML.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run the dot-tracking demo (prints metrics to stdout)
python scripts/run_dot_tracking.py --config configs/default.yaml
```

## Web UI

A tiny Streamlit UI lets you toggle features, launch runs, and plot results without the terminal.

```bash
pip install -r requirements.txt   # (contains streamlit + matplotlib)
streamlit run ui/app.py
```


Ablations:
```bash
# Disable rhythms
python scripts/run_dot_tracking.py --config configs/default.yaml rhythms.enabled=false

# Disable plasticity
python scripts/run_dot_tracking.py --config configs/default.yaml plasticity.enabled=false

# Disable router (static wiring)
python scripts/run_dot_tracking.py --config configs/default.yaml router.enabled=false
```

Artifacts: logs print to stdout; you can redirect to a file. The demo is intentionally lightweight.
