# cog_fabric/logging_utils/csvlogger.py
import csv, time
from pathlib import Path

class CSVLogger:
    def __init__(self, out_dir=None, fname=None):
        # repo root = .../cog_fabric (two levels up from this file)
        repo_root = Path(__file__).resolve().parents[2]
        runs_dir = (Path(out_dir) if out_dir else (repo_root / "runs"))
        runs_dir.mkdir(parents=True, exist_ok=True)

        ts = time.strftime("%Y%m%d-%H%M%S")
        name = fname or f"metrics_{ts}.csv"
        self.path = runs_dir / name

        self._file = open(self.path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow([
            "global_step","episode","t_in_ep",
            "loss","mode","theta_phase","gamma_phase",
            "perturb","router_err_mean"
        ])
        self._file.flush()

    def write(self, **row):
        self._writer.writerow([
            row.get("global_step"), row.get("episode"), row.get("t_in_ep"),
            row.get("loss"), row.get("mode"), row.get("theta_phase"),
            row.get("gamma_phase"), row.get("perturb"),
            row.get("router_err_mean")
        ])
        self._file.flush()

    def close(self):
        try: self._file.close()
        except Exception: pass
