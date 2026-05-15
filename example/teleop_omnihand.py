"""Teleoperation / dry-run entry point for OmniHand 2025.

This script reuses the Retargeter interface and outputs OmniHand 10 active
joint angles. It defaults to dry-run mode so the retargeting pipeline can be
validated without hardware or the OmniHand Python SDK installed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from input_devices.mediapipe_replay import MediaPipeReplay


OMNIHAND_VELOCITY_LIMITS = np.array(
    [0.164, 0.164, 0.308, 0.164, 0.308, 0.308, 0.164, 0.308, 0.164, 0.308],
    dtype=np.float64,
)


class OmniHandSafetyFilter:
    """Clamp, rate-limit, and smooth OmniHand active joint commands."""

    def __init__(
        self,
        joint_limits: np.ndarray,
        velocity_limits: np.ndarray = OMNIHAND_VELOCITY_LIMITS,
        alpha: float = 0.25,
        initial_qpos: np.ndarray | None = None,
    ):
        self.joint_limits = np.asarray(joint_limits, dtype=np.float64)
        self.velocity_limits = np.asarray(velocity_limits, dtype=np.float64)
        self.alpha = float(alpha)
        if self.joint_limits.shape != (10, 2):
            raise ValueError(f"Expected joint_limits shape (10, 2), got {self.joint_limits.shape}")
        if self.velocity_limits.shape != (10,):
            raise ValueError(f"Expected velocity_limits shape (10,), got {self.velocity_limits.shape}")
        if not (0.0 < self.alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {self.alpha}")

        self.q_prev = None
        self.t_prev = None
        if initial_qpos is not None:
            self.reset(initial_qpos)

    def reset(self, initial_qpos: np.ndarray | None = None, timestamp: float | None = None):
        if initial_qpos is None:
            self.q_prev = None
            self.t_prev = None
            return

        initial_qpos = np.asarray(initial_qpos, dtype=np.float64)
        if initial_qpos.shape != (10,):
            raise ValueError(f"Expected initial_qpos shape (10,), got {initial_qpos.shape}")
        if not np.all(np.isfinite(initial_qpos)):
            initial_qpos = self.joint_limits.mean(axis=1)

        self.q_prev = np.clip(initial_qpos, self.joint_limits[:, 0], self.joint_limits[:, 1])
        self.t_prev = time.time() if timestamp is None else float(timestamp)

    def next(self, q_target: np.ndarray, timestamp: float | None = None) -> np.ndarray:
        q_target = np.asarray(q_target, dtype=np.float64)
        if q_target.shape != (10,):
            raise ValueError(f"Expected OmniHand q shape (10,), got {q_target.shape}")

        if not np.all(np.isfinite(q_target)):
            if self.q_prev is not None:
                return self.q_prev.copy()
            q_target = self.joint_limits.mean(axis=1)

        q_target = np.clip(q_target, self.joint_limits[:, 0], self.joint_limits[:, 1])

        now = time.time() if timestamp is None else float(timestamp)
        if self.q_prev is None:
            self.q_prev = q_target.copy()
            self.t_prev = now
            return self.q_prev.copy()

        dt = max(now - self.t_prev, 1e-3)
        max_step = self.velocity_limits * dt
        q_rate_limited = self.q_prev + np.clip(q_target - self.q_prev, -max_step, max_step)
        q_filtered = self.q_prev + self.alpha * (q_rate_limited - self.q_prev)
        q_filtered = np.clip(q_filtered, self.joint_limits[:, 0], self.joint_limits[:, 1])

        self.q_prev = q_filtered
        self.t_prev = now
        return q_filtered.copy()


def _default_config_for_hand(hand_side: str) -> str:
    return f"config/omnihand/vector_omnihand_{hand_side}.yaml"


def _create_omnihand(hand_side: str, device_id: int):
    try:
        from omnihand_2025 import AgibotHandO10, EHandType
    except ImportError as exc:
        raise ImportError(
            "OmniHand hardware mode requires the omnihand_2025 Python package. "
            "Run with --dry-run to validate retargeting without hardware."
        ) from exc

    hand_type = EHandType.RIGHT if hand_side == "right" else EHandType.LEFT
    return AgibotHandO10.create_hand(device_id=device_id, hand_type=hand_type)


def _read_initial_active_qpos(hand, fallback: np.ndarray) -> np.ndarray:
    if hand is None:
        return fallback.copy()

    try:
        qpos = np.asarray(hand.get_all_active_joint_angles(), dtype=np.float64)
    except Exception as exc:
        print(f"[warn] Could not read OmniHand current active joints: {exc}. Using mid-range start.")
        return fallback.copy()

    if qpos.shape != (10,) or not np.all(np.isfinite(qpos)):
        print("[warn] Invalid OmniHand current active joints. Using mid-range start.")
        return fallback.copy()
    return qpos


def _create_retargeter(config_file: Path, hand_side: str):
    try:
        from wuji_retargeting import Retargeter
    except ImportError as exc:
        raise ImportError(
            "OmniHand retargeting requires the project runtime dependencies "
            "(notably nlopt and pinocchio). Use Python >=3.10 and install the "
            "package dependencies before running this example."
        ) from exc

    return Retargeter.from_yaml(str(config_file), hand_side)


def run_omnihand(
    hand_side: str,
    config_path: str,
    replay_path: str,
    dry_run: bool,
    device_id: int,
    playback_speed: float,
    playback_loop: bool,
    max_frames: int,
    print_every: int,
):
    hand_side = hand_side.lower()
    assert hand_side in {"left", "right"}

    config_file = Path(__file__).parent / config_path
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    retargeter = _create_retargeter(config_file, hand_side)
    joint_limits = retargeter.optimizer.robot.joint_limits

    input_device = MediaPipeReplay(
        record_path=replay_path,
        playback_speed=playback_speed,
        loop=playback_loop,
    )

    hand = None if dry_run else _create_omnihand(hand_side, device_id)
    initial_qpos = _read_initial_active_qpos(hand, joint_limits.mean(axis=1))
    safety = OmniHandSafetyFilter(joint_limits, initial_qpos=initial_qpos)

    frame_count = 0
    start_time = time.time()
    try:
        while max_frames <= 0 or frame_count < max_frames:
            fingers_data = input_device.get_fingers_data()
            fingers_pose = fingers_data[f"{hand_side}_fingers"]

            if np.allclose(fingers_pose, 0):
                if hasattr(input_device, "is_finished") and input_device.is_finished():
                    break
                time.sleep(0.01)
                continue

            q_raw, verbose = retargeter.retarget_verbose(fingers_pose, apply_filter=False)
            q_cmd = safety.next(q_raw)

            if hand is not None:
                hand.set_all_active_joint_angles(q_cmd.tolist())

            frame_count += 1
            if print_every > 0 and (frame_count == 1 or frame_count % print_every == 0):
                elapsed = max(time.time() - start_time, 1e-6)
                fps = frame_count / elapsed
                print(
                    f"frame={frame_count:05d} fps={fps:5.1f} "
                    f"cost={verbose['cost']:.4f} q_active={np.round(q_cmd, 4).tolist()}"
                )

            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\nStopping OmniHand retargeting...")


def main():
    parser = argparse.ArgumentParser(description="OmniHand 2025 retargeting")
    parser.add_argument("--hand", choices=["left", "right"], default="right")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--play", type=str, default="data/avp1.pkl")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--no-loop", action="store_true")
    parser.add_argument("--frames", type=int, default=300, help="0 means run forever")
    parser.add_argument("--print-every", type=int, default=30)
    parser.add_argument("--device-id", type=int, default=1)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry-run is the default; kept as an explicit flag for scripts.",
    )
    parser.add_argument(
        "--hardware",
        action="store_true",
        help="Connect to OmniHand hardware instead of dry-run.",
    )
    args = parser.parse_args()

    if sys.version_info < (3, 10):
        found = ".".join(str(part) for part in sys.version_info[:3])
        print(
            f"[blocked] Python >=3.10 is required by this project, but this interpreter is {found}.",
            file=sys.stderr,
        )
        return 2

    dry_run = not args.hardware
    config_path = args.config or _default_config_for_hand(args.hand)

    try:
        run_omnihand(
            hand_side=args.hand,
            config_path=config_path,
            replay_path=args.play,
            dry_run=dry_run,
            device_id=args.device_id,
            playback_speed=args.speed,
            playback_loop=not args.no_loop,
            max_frames=args.frames,
            print_every=args.print_every,
        )
    except (FileNotFoundError, ImportError, ValueError) as exc:
        print(f"[blocked] {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
