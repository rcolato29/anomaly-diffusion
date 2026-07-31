from anomaly_diffusion.serving.drift import DriftMonitor


def test_no_drift_score_until_window_full():
    mon = DriftMonitor(ref_mean=1.0, ref_std=0.5, window=10)
    mon.update([1.0] * 5)
    assert mon.drift_z() is None  # not warm yet
    assert mon.status()["drifting"] is None


def test_matching_distribution_has_low_drift():
    mon = DriftMonitor(ref_mean=1.0, ref_std=0.5, window=10, alert_z=3.0)
    mon.update([1.0] * 10)  # live mean == ref mean
    assert abs(mon.drift_z()) < 1e-6
    assert mon.status()["drifting"] is False


def test_shifted_distribution_flags_drift():
    mon = DriftMonitor(ref_mean=1.0, ref_std=0.5, window=10, alert_z=3.0)
    mon.update([3.0] * 10)  # +2.0 shift = 4 std -> drifting
    assert mon.drift_z() > 3.0
    assert mon.status()["drifting"] is True


def test_from_reference_computes_stats():
    mon = DriftMonitor.from_reference([1.0, 2.0, 3.0], window=3)
    assert mon.ref_mean == 2.0
    assert mon.ref_std > 0
