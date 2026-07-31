"""Model-native input-drift monitor.

The diffusion model is already a density model of normal data, so its anomaly score doubles
as a drift signal. The score distribution on a reference window of known-good normals is
summarized once, and a rolling live window is compared against it.

Drift is a distributional shift across inputs, distinct from a single defective image. A
lighting or focus change raises the score of normals across the board, moving the live-window
mean even when no individual image trips the anomaly threshold. The reported statistic is
that mean's shift from the reference mean, in units of the reference standard deviation. A
sustained large value calls for recalibrating the EVT threshold or retraining.
"""

from __future__ import annotations

import statistics
from collections import deque


class DriftMonitor:
    def __init__(self, ref_mean: float, ref_std: float, window: int = 50, alert_z: float = 3.0):
        self.ref_mean = ref_mean
        self.ref_std = max(ref_std, 1e-8)
        self.window = window
        self.alert_z = alert_z
        self._live: deque[float] = deque(maxlen=window)

    @classmethod
    def from_reference(
        cls, reference_scores, window: int = 50, alert_z: float = 3.0
    ) -> DriftMonitor:
        return cls(
            ref_mean=statistics.fmean(reference_scores),
            ref_std=statistics.pstdev(reference_scores) if len(reference_scores) > 1 else 0.0,
            window=window,
            alert_z=alert_z,
        )

    def update(self, scores) -> None:
        for s in scores:
            self._live.append(float(s))

    def drift_z(self) -> float | None:
        """Live-window mean shift from reference, in reference-std units. None until warm."""
        if len(self._live) < self.window:
            return None
        return (statistics.fmean(self._live) - self.ref_mean) / self.ref_std

    def status(self) -> dict:
        z = self.drift_z()
        return {
            "drift_z": z,
            "drifting": None if z is None else bool(abs(z) >= self.alert_z),
            "live_n": len(self._live),
            "window": self.window,
        }
