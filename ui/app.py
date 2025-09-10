# ui/app.py
import subprocess, sys, os, time, glob, yaml, csv
from pathlib import Path
import streamlit as st
import matplotlib.pyplot as plt
import math

def linspace(a, b, n):
    if n <= 1: 
        return [a]
    step = (b - a) / (n - 1)
    return [a + i * step for i in range(n)]

def run_one_and_summarize(overrides):
    path = run_experiment(overrides)
    if not path:
        return None, None, None
    rows = load_csv_rows(path)
    mean_err, hl = compute_summary(rows)
    return str(path), mean_err, hl


def list_csvs():
    RUNS_DIR.mkdir(exist_ok=True, parents=True)
    return sorted(RUNS_DIR.glob("metrics_*.csv"))

def load_csv_rows(path):
    import csv
    rows = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def compute_summary(rows):
    # mean loss
    losses = [float(r["loss"]) for r in rows]
    mean_err = sum(losses) / max(1, len(losses))
    # crude half-life: find first big spike (e.g., 90th percentile), then steps to reach half of that
    if not losses:
        return mean_err, None
    import numpy as np
    arr = np.array(losses, dtype=float)
    spike = float(np.percentile(arr, 90))
    target = spike * 0.5
    idx = int(np.argmax(arr))  # first max location
    hl = None
    for j in range(idx + 1, len(arr)):
        if arr[j] <= target:
            hl = j - idx
            break
    return mean_err, hl


REPO_ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = REPO_ROOT / "configs" / "default.yaml"
RUN_SCRIPT = ["python", "-m", "scripts.run_dot_tracking", "--config", str(CFG_PATH)]
RUNS_DIR = REPO_ROOT / "runs"

def load_cfg():
    with open(CFG_PATH, "r") as f:
        return yaml.safe_load(f)

def find_latest_csv():
    RUNS_DIR.mkdir(exist_ok=True, parents=True)
    files = sorted(RUNS_DIR.glob("metrics_*.csv"))
    return files[-1] if files else None

def run_experiment(overrides: dict):
    # Build dotpath=value overrides
    args = RUN_SCRIPT.copy()
    for k, v in overrides.items():
        args.append(f"{k}={v}")
    env = os.environ.copy()
    # ensure repo root on PYTHONPATH
    env["PYTHONPATH"] = str(REPO_ROOT)
    st.write("Launching:", " ".join(args))
    proc = subprocess.Popen(args, cwd=str(REPO_ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    # stream logs live
    log = st.empty()
    buf = []
    while True:
        line = proc.stdout.readline()
        if not line and proc.poll() is not None:
            break
        if line:
            buf.append(line.rstrip())
            if len(buf) > 300: buf = buf[-300:]
            log.text("\n".join(buf))
    code = proc.wait()
    st.success(f"Run finished (exit {code})")
    return find_latest_csv()

def plot_csv(path, mark_shock=False):
    xs, ys, shocks = [], [], []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            xs.append(int(row["global_step"]))
            ys.append(float(row["loss"]))
            try:
                if int(row.get("perturb", 0)) == 1:
                    shocks.append(int(row["global_step"]))
            except Exception:
                pass
    fig = plt.figure()
    plt.plot(xs, ys, label=os.path.basename(path))
    if mark_shock and shocks:
        plt.axvline(shocks[0], linestyle="--")
    plt.xlabel("Global step")
    plt.ylabel("Loss")
    plt.title("Tracking loss over time")
    plt.legend()
    st.pyplot(fig)

st.set_page_config(page_title="Cog Fabric UI", layout="wide")
st.title("🧠 Cog Fabric — Run & Visualize")

cfg = load_cfg()

with st.sidebar:
    st.header("Features")
    rhythms = st.toggle("Rhythms enabled", value=cfg["rhythms"]["enabled"])
    plastic = st.toggle("Plasticity enabled", value=cfg["plasticity"]["enabled"])
    router  = st.toggle("Router enabled",     value=cfg["router"]["enabled"])
    memory  = st.toggle("Memory enabled",     value=cfg["memory"]["enabled"])
    st.divider()

    st.header("Rhythm knobs")
    theta_hz = st.slider("theta_hz", 1.0, 10.0, float(cfg["rhythms"]["theta_hz"]), 0.5)
    gamma_hz = st.slider("gamma_hz", 20.0, 120.0, float(cfg["rhythms"]["gamma_hz"]), 5.0)
    theta_gate = st.slider("theta_gate_frac", 0.05, 0.5, float(cfg["rhythms"]["theta_gate_frac"]), 0.01)
    gamma_gate = st.slider("gamma_gate_frac", 0.1, 1.0, float(cfg["rhythms"]["gamma_gate_frac"]), 0.05)
    st.divider()

    st.header("Training")
    steps = st.number_input("steps_per_episode", 50, 2000, int(cfg["env"]["steps_per_episode"]), 50)
    episodes = st.number_input("episodes", 1, 100, int(cfg["env"]["episodes"]), 1)
    sleep_epochs = st.number_input("sleep_replay_epochs", 0, 10, int(cfg["training"]["sleep_replay_epochs"]), 1)
    seed = st.number_input("seed", 0, 999999, int(cfg["training"]["seed"]), 1)
    st.divider()

    st.header("Perturbation")
    pert_enabled = st.toggle("Enable shock", value=bool(cfg.get("perturb", {}).get("enabled", False)))
    pert_step = st.number_input("step", 1, 2000, int(cfg.get("perturb", {}).get("step", 200)), 1)
    pert_mult = st.number_input("speed_multiplier", 1.0, 8.0, float(cfg.get("perturb", {}).get("speed_multiplier", 2.0)), 0.1)
    noise_std = st.number_input("noise_std", 0.0, 1.0, float(cfg.get("perturb", {}).get("noise_std", 0.0)), 0.01)
    noise_steps = st.number_input("noise_steps", 0, 200, int(cfg.get("perturb", {}).get("noise_steps", 0)), 1)

col1, col2 = st.columns([1,1])

with col1:
    if st.button("▶ Run experiment"):
        overrides = {
            "rhythms.enabled": rhythms,
            "plasticity.enabled": plastic,
            "router.enabled": router,
            "memory.enabled": memory,
            "rhythms.theta_hz": theta_hz,
            "rhythms.gamma_hz": gamma_hz,
            "rhythms.theta_gate_frac": theta_gate,
            "rhythms.gamma_gate_frac": gamma_gate,
            "env.steps_per_episode": int(steps),
            "env.episodes": int(episodes),
            "training.sleep_replay_epochs": int(sleep_epochs),
            "training.seed": int(seed),
            "perturb.enabled": pert_enabled,
            "perturb.step": int(pert_step),
            "perturb.speed_multiplier": float(pert_mult),
            "perturb.noise_std": float(noise_std),
            "perturb.noise_steps": int(noise_steps),
        }
        csv_path = run_experiment(overrides)
        if csv_path:
            st.session_state["last_csv"] = str(csv_path)

with col2:
    st.subheader("Compare runs (overlay)")
    all_csvs = [str(p) for p in list_csvs()]
    default_sel = []
    # auto-select the latest one
    if all_csvs:
        default_sel = [all_csvs[-1]]
    chosen = st.multiselect("Pick one or more runs to overlay", all_csvs, default=default_sel)

    if chosen:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        for path in chosen:
            xs, ys, shocks = [], [], []
            rows = load_csv_rows(path)
            for r in rows:
                xs.append(int(r["global_step"]))
                ys.append(float(r["loss"]))
                try:
                    if int(r.get("perturb", 0)) == 1:
                        shocks.append(int(r["global_step"]))
                except Exception:
                    pass
            plt.plot(xs, ys, label=os.path.basename(path))
            if shocks:
                plt.axvline(shocks[0], linestyle="--", alpha=0.6)
        plt.xlabel("Global step")
        plt.ylabel("Loss")
        plt.title("Tracking loss over time (overlay)")
        plt.legend(fontsize="small")
        plt.tight_layout()
        st.pyplot(fig)

        # quick per-file summaries
        st.markdown("**Summaries:**")
        for path in chosen:
            rows = load_csv_rows(path)
            mean_err, hl = compute_summary(rows)
            st.write(f"- `{os.path.basename(path)}` → mean_err≈{mean_err:.4f}"
                     + (f", half-life≈{hl} steps" if hl is not None else ""))
    else:
        st.info("Select one or more CSVs from `runs/` to compare.")

st.markdown("---")
st.header("🔎 Sweep mode — find the sweet spot")

# Choose a parameter to sweep (common useful ones)
param_choices = [
    "rhythms.theta_gate_frac",
    "rhythms.theta_hz",
    "rhythms.gamma_hz",
    "reservoir.spectral_radius",
    "reservoir.leak_rate",
    "router.error_threshold",
]
sweep_param = st.selectbox("Parameter", options=param_choices, index=0)

# Ranges (float) and number of points
c1, c2, c3 = st.columns([1,1,1])
with c1:
    sweep_min = st.number_input("min", value=0.05, step=0.01, format="%.4f")
with c2:
    sweep_max = st.number_input("max", value=0.50, step=0.01, format="%.4f")
with c3:
    sweep_points = st.number_input("# points", min_value=2, max_value=50, value=8, step=1)

# Keep other controls from sidebar as the base config for the sweep
base_overrides = {
    "rhythms.enabled": rhythms,
    "plasticity.enabled": plastic,
    "router.enabled": router,
    "memory.enabled": memory,
    "rhythms.theta_hz": theta_hz,
    "rhythms.gamma_hz": gamma_hz,
    "rhythms.theta_gate_frac": theta_gate,
    "rhythms.gamma_gate_frac": gamma_gate,
    "env.steps_per_episode": int(steps),
    "env.episodes": int(episodes),
    "training.sleep_replay_epochs": int(sleep_epochs),
    "training.seed": int(seed),
    "perturb.enabled": pert_enabled,
    "perturb.step": int(pert_step),
    "perturb.speed_multiplier": float(pert_mult),
    "perturb.noise_std": float(noise_std),
    "perturb.noise_steps": int(noise_steps),
}

if st.button("🚀 Run sweep"):
    values = linspace(float(sweep_min), float(sweep_max), int(sweep_points))
    results = []  # list of (param_value, mean_err, half_life, csv_path)

    prog = st.progress(0.0, text="Starting sweep…")
    for i, val in enumerate(values, start=1):
        ov = dict(base_overrides)
        ov[sweep_param] = float(val)
        st.write(f"Running {sweep_param}={val:.5f}")
        csv_path, mean_err, hl = run_one_and_summarize(ov)
        results.append((float(val), mean_err, hl, csv_path))
        prog.progress(i / len(values), text=f"{i}/{len(values)} done")

    # Results table
    st.subheader("Sweep results")
    import pandas as pd
    df = pd.DataFrame(results, columns=[sweep_param, "mean_err", "half_life", "csv"])
    st.dataframe(df, use_container_width=True)

    # Plot half-life vs parameter (and mean_err as optional)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    xs = [r[0] for r in results]
    hls = [r[2] if r[2] is not None else math.nan for r in results]
    plt.plot(xs, hls, marker="o")
    plt.xlabel(sweep_param)
    plt.ylabel("Adaptation half-life (steps)")
    plt.title("Sweep: half-life vs parameter")
    plt.tight_layout()
    st.pyplot(fig)

    # Save CSV in runs/
    out_csv = RUNS_DIR / f"sweep_{sweep_param.replace('.', '_')}.csv"
    df.to_csv(out_csv, index=False)
    st.success(f"Saved sweep CSV → {out_csv}")
