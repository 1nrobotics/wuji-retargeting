"""Validate OmniHand retargeting setup.

This script is intentionally hardware-free. It checks configuration files,
runtime dependencies, OmniHand FK/Jacobian, and optionally one retargeting frame
from a replay file. Use it before connecting real hardware.
"""

from __future__ import annotations

import argparse
import importlib
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _python_version_error() -> str | None:
    if sys.version_info >= (3, 10):
        return None

    found = ".".join(str(part) for part in sys.version_info[:3])
    return (
        f"Python >=3.10 is required by this project, but this interpreter is {found}. "
        "Create/use a Python 3.10+ environment before FK/Jacobian/retarget tests."
    )


def _check_imports() -> list[str]:
    missing = []
    for module in ("nlopt", "pinocchio", "scipy"):
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(module)
    return missing


def _load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _validate_config(path: Path):
    config = _load_yaml(path)
    robot = config.get("robot", {})
    optimizer = config.get("optimizer", {})
    retarget = config.get("retarget", {})
    key_vectors = retarget.get("key_vectors", [])

    assert robot.get("type") == "OmniHand", f"{path}: robot.type must be OmniHand"
    assert optimizer.get("type") == "VectorOptimizer", f"{path}: optimizer.type must be VectorOptimizer"
    assert len(key_vectors) == 15, f"{path}: expected 15 key vectors, got {len(key_vectors)}"

    for i, kv in enumerate(key_vectors):
        for key in ("origin", "task", "origin_kp", "task_kp", "scale"):
            assert key in kv, f"{path}: key_vectors[{i}] missing {key}"

    print(f"[ok] config {path} ({len(key_vectors)} key vectors)")


def _finite_difference_jacobian(robot, q, link_indices, eps=1e-6):
    base = robot.compute_fk_batch(q, link_indices).reshape(len(link_indices), 3)
    fd = np.zeros((len(link_indices), 3, robot.num_opt_joints), dtype=np.float64)
    for j in range(robot.num_opt_joints):
        dq = np.zeros_like(q)
        dq[j] = eps
        plus = robot.compute_fk_batch(q + dq, link_indices).reshape(len(link_indices), 3)
        minus = robot.compute_fk_batch(q - dq, link_indices).reshape(len(link_indices), 3)
        fd[:, :, j] = (plus - minus) / (2.0 * eps)
    return base, fd


def _validate_robot(hand_side: str):
    from wuji_retargeting.robot_omnihand import OmniHandRobotWrapper

    robot = OmniHandRobotWrapper(hand_side=hand_side)
    limits = robot.joint_limits
    q_mid = limits.mean(axis=1)
    q_full = robot.active_to_full(q_mid)

    assert q_mid.shape == (10,)
    assert q_full.shape == (16,)
    assert np.all(np.isfinite(q_full))

    prefix = "R" if hand_side == "right" else "L"
    tip_links = [
        f"{prefix}_thumb_tip",
        f"{prefix}_index_tip",
        f"{prefix}_middle_tip",
        f"{prefix}_ring_tip",
        f"{prefix}_pinky_tip",
    ]
    link_indices = [robot.get_link_index(name) for name in tip_links]
    fk = robot.compute_fk_batch(q_mid, link_indices)
    jac = robot.compute_all_jacobians_batch(q_mid, link_indices)

    assert fk.shape == (len(link_indices) * 3,)
    assert jac.shape == (len(link_indices), 3, 10)
    assert np.all(np.isfinite(fk))
    assert np.all(np.isfinite(jac))

    _, jac_fd = _finite_difference_jacobian(robot, q_mid, link_indices)
    max_err = float(np.max(np.abs(jac - jac_fd)))
    if max_err > 5e-3:
        raise AssertionError(f"{hand_side} Jacobian finite-difference error too high: {max_err:.6g}")

    print(
        f"[ok] {hand_side} robot nq={robot.model.nq} nv={robot.model.nv} "
        f"active=10 full=16 jac_fd_max_err={max_err:.3g}"
    )


def _validate_one_retarget_frame(hand_side: str, config_path: Path, replay_path: Path):
    from wuji_retargeting import Retargeter

    with open(replay_path, "rb") as f:
        frames = pickle.load(f)
    hand_key = f"{hand_side}_fingers"
    frame = next((item for item in frames if not np.allclose(item[hand_key], 0)), None)
    if frame is None:
        raise RuntimeError(f"No non-zero {hand_key} frame found in {replay_path}")

    retargeter = Retargeter.from_yaml(str(config_path), hand_side)
    qpos, verbose = retargeter.retarget_verbose(frame[hand_key], apply_filter=False)

    assert qpos.shape == (10,)
    assert np.all(np.isfinite(qpos))
    print(
        f"[ok] one-frame retarget {hand_side}: "
        f"q_shape={qpos.shape} cost={verbose['cost']:.4f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Validate OmniHand retargeting setup")
    parser.add_argument("--hand", choices=["left", "right", "both"], default="both")
    parser.add_argument("--skip-retarget", action="store_true")
    parser.add_argument("--play", type=str, default="data/avp1.pkl")
    args = parser.parse_args()

    config_dir = Path(__file__).parent / "config" / "omnihand"
    sides = ["left", "right"] if args.hand == "both" else [args.hand]

    for side in sides:
        _validate_config(config_dir / f"vector_omnihand_{side}.yaml")

    version_error = _python_version_error()
    if version_error:
        print(f"[blocked] {version_error}")
        return 2

    missing = _check_imports()
    if missing:
        print(
            "[blocked] Missing runtime dependencies: "
            + ", ".join(missing)
            + ". Install project dependencies before FK/Jacobian/retarget tests."
        )
        return 2

    for side in sides:
        _validate_robot(side)

    if not args.skip_retarget:
        replay_path = Path(__file__).parent / args.play
        for side in sides:
            _validate_one_retarget_frame(
                side,
                config_dir / f"vector_omnihand_{side}.yaml",
                replay_path,
            )

    print("[ok] OmniHand setup validation complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
