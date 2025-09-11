import argparse
import yaml
import numpy as np
import torch

from cog_fabric.envs.dot_env import EventDotEnv
from cog_fabric.rhythms.oscillator import RhythmGater
from cog_fabric.controller.agent import Agent
from cog_fabric.logging_utils.metrics import MetricTracker
from cog_fabric.logging_utils.csvlogger import CSVLogger


def parse_arg_overrides(overrides):
    """
    Parse CLI overrides like: key.subkey=value key2=3 key3=true
    Returns a flat dict of { "key.subkey": parsed_value }
    """
    out = {}
    for ov in overrides:
        if "=" not in ov:
            continue
        k, v = ov.split("=", 1)
        vl = v.strip()
        if vl.lower() in ("true", "false"):
            out[k] = (vl.lower() == "true")
        else:
            try:
                # ints first, then floats
                if vl.isdigit() or (vl.startswith("-") and vl[1:].isdigit()):
                    out[k] = int(vl)
                else:
                    out[k] = float(vl)
            except ValueError:
                out[k] = vl
    return out


def apply_overrides(cfg, flat):
    """Apply dot-path overrides into nested dict."""
    for k, v in flat.items():
        parts = k.split(".")
        cur = cfg
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = v
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("overrides", nargs="*", help="dotpath=value (e.g., rhythms.enabled=false)")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    cfg = apply_overrides(cfg, parse_arg_overrides(args.overrides))

    # Ensure decimal floats where needed (YAML may parse 1e-3 as str)
    # Safe-guard key fields if present
    try:
        cfg["reservoir"]["lr"] = float(cfg["reservoir"]["lr"])
        cfg["plasticity"]["lr"] = float(cfg["plasticity"]["lr"])
        cfg["plasticity"]["decay"] = float(cfg["plasticity"]["decay"])
    except Exception:
        pass

    torch.manual_seed(cfg["training"]["seed"])
    np.random.seed(cfg["training"]["seed"])

    env = EventDotEnv(
        width=cfg["env"]["width"],
        height=cfg["env"]["height"],
        event_threshold=cfg["env"]["event_threshold"],
        max_speed=cfg["env"]["max_speed"],
        seed=cfg["training"]["seed"],
        perturb=cfg.get("perturb", {"enabled": False}),
    )

    features = dict(
        rhythms_enabled=cfg["rhythms"]["enabled"],
        plasticity_enabled=cfg["plasticity"]["enabled"],
        memory_enabled=cfg["memory"]["enabled"],
        router_enabled=cfg["router"]["enabled"],
    )

    # Rhythm generator (even if disabled, we’ll stub phases as 0.0)
    rhythm = RhythmGater(
        theta_hz=cfg["rhythms"]["theta_hz"],
        gamma_hz=cfg["rhythms"]["gamma_hz"],
        theta_gate_frac=cfg["rhythms"]["theta_gate_frac"],
        gamma_gate_frac=cfg["rhythms"]["gamma_gate_frac"],
        dt=0.01,
    )

    agent = Agent(env_shape=(2, cfg["env"]["height"], cfg["env"]["width"]), cfg=cfg, device="cpu")
    metrics = MetricTracker()
    csvlog = CSVLogger(out_dir="runs")

    episodes = int(cfg["env"]["episodes"])
    steps_per_episode = int(cfg["env"]["steps_per_episode"])

    print("CONFIG:", cfg)
    print("FEATURES:", features)
    if cfg.get("perturb", {}).get("enabled", False):
        p = cfg["perturb"]
        print(
            f"[perturb] enabled at t={int(p.get('step', 200))} "
            f"x{float(p.get('speed_multiplier', 2.0))} "
            f"noise_std={float(p.get('noise_std', 0.0))} "
            f"noise_steps={int(p.get('noise_steps', 0))}"
        )

    for ep in range(episodes):
        ev, target = env.reset()
        agent.reset()
        print(f"\n=== Episode {ep + 1}/{episodes} ===")
        for t in range(steps_per_episode):
            if features["rhythms_enabled"]:
                plasticity_gate, routing_gate, theta_phase, gamma_phase = rhythm.step()
            else:
                plasticity_gate, routing_gate, theta_phase, gamma_phase = (True, True, 0.0, 0.0)

            loss = agent.step(
                ev,
                target,
                gates=(plasticity_gate, routing_gate, theta_phase, gamma_phase),
                features=features,
            )
            metrics.log(step_idx=ep * steps_per_episode + t, err=loss, mode="train")

            # write one CSV row
            csvlog.write(
                global_step=ep * steps_per_episode + t,
                episode=ep,
                t_in_ep=t,
                loss=loss,
                mode="train",
                theta_phase=theta_phase,
                gamma_phase=gamma_phase,
                perturb=int(
                    cfg.get("perturb", {}).get("enabled", False)
                    and t == int(cfg.get("perturb", {}).get("step", 200))
                ),
                router_err_mean=agent.router.err_mean if features["router_enabled"] else 0.0,
            )

            # step environment
            ev, target = env.step()

            if (t + 1) % 100 == 0:
                print(f"t={t + 1:4d}  loss={loss:.4f}  mean_err={metrics.mean_err():.4f}")

        # sleep/replay between episodes
        if int(cfg["training"]["sleep_replay_epochs"]) > 0:
            avg = agent.sleep_replay(epochs=int(cfg["training"]["sleep_replay_epochs"]))
            print(f"[sleep] replay avg loss: {avg:.4f}")

    hl = metrics.adaptation_half_life()
    print("\n=== Summary ===")
    print(f"Mean error: {metrics.mean_err():.4f}")
    print(f"Adaptation half-life (steps): {hl}")
    csvlog.close()
    print(f"CSV written to: {csvlog.path}")


if __name__ == "__main__":
    main()
