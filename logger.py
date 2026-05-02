"""
Training metrics: CSV logs under ``runs/<name>_<timestamp>/`` and a loss curve PNG.

Usage::

    log = TrainingLogger.create(name="jepa")
    ...
    log.log_epoch(epoch, avg_loss, wall_time_s=...)
    log.log_step({"step": step, "epoch": epoch, "loss": loss, ...})  # optional
    log.close()
"""

from __future__ import annotations

import csv
from contextlib import AbstractContextManager
from datetime import datetime
from pathlib import Path
from typing import Any


class TrainingLogger(AbstractContextManager["TrainingLogger"]):
    def __enter__(self) -> TrainingLogger:
        return self

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._epoch_rows: list[tuple[int, float, float | None]] = []
        self._epochs_fp: Any = None
        self._epochs_writer: csv.writer | None = None
        self._steps_fp: Any = None
        self._steps_writer: csv.DictWriter | None = None
        self._steps_fields: list[str] | None = None

    @property
    def run_path(self) -> Path:
        return self.run_dir

    @classmethod
    def create(cls, base_dir: str | Path = "runs", name: str = "run") -> TrainingLogger:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(base_dir) / f"{name}_{ts}"
        log = cls(run_dir)
        log._open_epochs_csv()
        return log

    def _open_epochs_csv(self) -> None:
        path = self.run_dir / "epochs.csv"
        self._epochs_fp = open(path, "w", newline="")
        self._epochs_writer = csv.writer(self._epochs_fp)
        self._epochs_writer.writerow(["epoch", "avg_loss", "wall_time_s"])

    def log_epoch(self, epoch: int, avg_loss: float, wall_time_s: float | None = None) -> None:
        assert self._epochs_writer is not None and self._epochs_fp is not None
        self._epoch_rows.append((epoch, avg_loss, wall_time_s))
        self._epochs_writer.writerow(
            [
                epoch,
                f"{avg_loss:.8f}",
                "" if wall_time_s is None else f"{wall_time_s:.2f}",
            ]
        )
        self._epochs_fp.flush()
        self._save_loss_plot()

    def log_step(self, row: dict[str, Any]) -> None:
        """Append one row to ``steps.csv`` (headers from the first row)."""
        if not row:
            return
        keys = [str(k) for k in row.keys()]
        if self._steps_fp is None:
            self._steps_fields = keys
            path = self.run_dir / "steps.csv"
            self._steps_fp = open(path, "w", newline="")
            self._steps_writer = csv.DictWriter(self._steps_fp, fieldnames=self._steps_fields)
            self._steps_writer.writeheader()
        assert self._steps_writer is not None and self._steps_fp is not None
        if keys != self._steps_fields:
            raise ValueError(
                f"log_step keys must match first row; expected {self._steps_fields}, got {keys}"
            )
        self._steps_writer.writerow({k: row[k] for k in self._steps_fields})
        self._steps_fp.flush()

    def _save_loss_plot(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self._epoch_rows:
            return
        epochs = [r[0] for r in self._epoch_rows]
        losses = [r[1] for r in self._epoch_rows]
        plt.figure(figsize=(8, 4))
        plt.plot(epochs, losses, marker="o", linewidth=1.5)
        plt.xlabel("Epoch")
        plt.ylabel("Average loss")
        plt.title("Training loss (per epoch)")
        plt.grid(True, alpha=0.35)
        plt.tight_layout()
        out = self.run_dir / "loss.png"
        plt.savefig(out, dpi=120)
        plt.close()

    def close(self) -> None:
        if self._epochs_fp is not None:
            self._epochs_fp.close()
            self._epochs_fp = None
            self._epochs_writer = None
        if self._steps_fp is not None:
            self._steps_fp.close()
            self._steps_fp = None
            self._steps_writer = None

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
