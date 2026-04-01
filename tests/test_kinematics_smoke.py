import numpy as np

from strandbeest.config import LinkageConfig
from strandbeest.kinematics import evaluate_design, solve_pose
from strandbeest.reference import REFERENCE


def test_reference_dimension_is_11():
    cfg = LinkageConfig()
    assert cfg.dim_x == 11
    assert len(REFERENCE.optimized_names) == 11


def test_canonical_pose_is_valid():
    cfg = LinkageConfig()
    pose = solve_pose(REFERENCE.canonical_array(), theta_deg=210.0, cfg=cfg)
    assert pose.valid
    assert "S" in pose.points
    assert np.isfinite(pose.points["S"]).all()


def test_canonical_design_produces_finite_metrics():
    cfg = LinkageConfig()
    res = evaluate_design(REFERENCE.canonical_array(), cfg)
    assert res.feasible
    assert res.trajectory.shape == (cfg.angle_samples, 2)
    assert res.metrics.shape == (4,)
    assert np.isfinite(res.metrics).all()

