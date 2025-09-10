import argparse, time, yaml, numpy as np, torch

from cog_fabric.envs.dot_env import EventDotEnv
from cog_fabric.rhythms.oscillator import RhythmGater
from cog_fabric.controller.agent import Agent
from cog_fabric.logging_utils.metrics import MetricTracker

def parse_arg_overrides(overrides):
    # simple dotpath=val parser for a few flags
    as_dict = {}
    for ov in overrides:
        if '=' not in ov:
            continue
        k, v = ov.split('=', 1)
        # interpret booleans/numbers
        if v.lower() in ('true','false'):
            val = v.lower() == 'true'
        else:
            try:
                val = float(v) if '.' in v else int(v)
            except:
                val = v
        as_dict[k] = val
    return as_dict

def apply_overrides(cfg, overrides):
    for k, v in overrides.items():
        parts = k.split('.')
        cur = cfg
        for p in parts[:-1]:
            if p not in cur:
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = v
    return cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('overrides', nargs='*', help='dotpath=value for quick edits')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    cfg = apply_overrides(cfg, parse_arg_overrides(args.overrides))

    torch.manual_seed(cfg['training']['seed'])
    np.random.seed(cfg['training']['seed'])

    env = EventDotEnv(width=cfg['env']['width'],
                      height=cfg['env']['height'],
                      event_threshold=cfg['env']['event_threshold'],
                      max_speed=cfg['env']['max_speed'],
                      seed=cfg['training']['seed'])

    rhythm = RhythmGater(theta_hz=cfg['rhythms']['theta_hz'],
                         gamma_hz=cfg['rhythms']['gamma_hz'],
                         theta_gate_frac=cfg['rhythms']['theta_gate_frac'],
                         gamma_gate_frac=cfg['rhythms']['gamma_gate_frac'],
                         dt=0.01)

    agent = Agent(env_shape=(2, cfg['env']['height'], cfg['env']['width']), cfg=cfg, device='cpu')
    metrics = MetricTracker()

    episodes = cfg['env']['episodes']
    steps_per_episode = cfg['env']['steps_per_episode']

    features = dict(
        rhythms_enabled = cfg['rhythms']['enabled'],
        plasticity_enabled = cfg['plasticity']['enabled'],
        memory_enabled = cfg['memory']['enabled'],
        router_enabled = cfg['router']['enabled'],
    )

    print("CONFIG:", cfg)
    print("FEATURES:", features)

    for ep in range(episodes):
        ev, target = env.reset()
        agent.reset()
        print(f"\n=== Episode {ep+1}/{episodes} ===")
        for t in range(steps_per_episode):
            if features['rhythms_enabled']:
                gates = rhythm.step()
            else:
                gates = (True, True, 0.0, 0.0)

            loss = agent.step(ev, target, gates, features)
            metrics.log(step_idx=ep*steps_per_episode + t, err=loss, mode='train')

            ev, target = env.step()

            if (t+1) % 100 == 0:
                print(f"t={t+1:4d}  loss={loss:.4f}  mean_err={metrics.mean_err():.4f}")

        # Sleep/replay between episodes
        if cfg['training']['sleep_replay_epochs'] > 0:
            avg = agent.sleep_replay(epochs=cfg['training']['sleep_replay_epochs'])
            print(f"[sleep] replay avg loss: {avg:.4f}")

    hl = metrics.adaptation_half_life()
    print("\n=== Summary ===")
    print(f"Mean error: {metrics.mean_err():.4f}")
    print(f"Adaptation half-life (steps): {hl}")

if __name__ == '__main__':
    main()
