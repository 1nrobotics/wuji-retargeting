"""Teleoperation with MuJoCo visualization for OmniHand 2025.

This viewer mirrors the Wuji `teleop_sim.py` flow, but OmniHand has no MuJoCo
actuators in the SDK URDF. The script therefore visualizes retargeting by
writing the retargeted active joints, plus derived passive joints, directly to
MuJoCo `data.qpos`.

Usage:
    # Replay MediaPipe recording with right OmniHand
    mjpython teleop_omnihand_mujoco.py --hand right --play data/avp1.pkl

    # MP4 video input with MediaPipe hand detection
    mjpython teleop_omnihand_mujoco.py --hand right --video data/right.mp4 --show-video

    # Live cameras
    mjpython teleop_omnihand_mujoco.py --hand right --realsense
    mjpython teleop_omnihand_mujoco.py --hand right --zed
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXAMPLE_ROOT = Path(__file__).resolve().parent
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from input_devices.mediapipe_replay import MediaPipeReplay
from teleop_omnihand import OmniHandSafetyFilter


def _python_version_error() -> str | None:
    if sys.version_info >= (3, 10):
        return None

    found = ".".join(str(part) for part in sys.version_info[:3])
    return f"Python >=3.10 is required by this project, but this interpreter is {found}."


def _import_mujoco():
    try:
        mujoco = importlib.import_module("mujoco")
        importlib.import_module("mujoco.viewer")
    except ImportError as exc:
        raise ImportError(
            "OmniHand MuJoCo visualization requires mujoco. Install project "
            "requirements and run with mjpython when using the native viewer."
        ) from exc
    return mujoco


def _create_retargeter(config_file: Path, hand_side: str):
    try:
        from wuji_retargeting import Retargeter
    except ImportError as exc:
        raise ImportError(
            "OmniHand retargeting requires nlopt and pinocchio. Use Python >=3.10 "
            "and install the project dependencies before running visualization."
        ) from exc

    return Retargeter.from_yaml(str(config_file), hand_side)


def _default_config_for_hand(hand_side: str) -> str:
    return f"config/omnihand/vector_omnihand_{hand_side}.yaml"


def _resolve_example_path(path: str | None) -> Path | None:
    if path is None:
        return None
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = EXAMPLE_ROOT / resolved
    return resolved


def _load_video_config(config_file: Path) -> dict:
    with open(config_file, "r") as f:
        config = yaml.safe_load(f) or {}
    return config.get("video_input", {})


def _create_input_device(
    input_device_type: str,
    hand_side: str,
    config_file: Path,
    replay_path: str,
    video_path: str,
    visionpro_ip: str,
    playback_speed: float,
    playback_loop: bool,
    show_video: bool,
):
    video_config = _load_video_config(config_file)

    if input_device_type == "mediapipe_replay":
        return MediaPipeReplay(
            record_path=replay_path,
            playback_speed=playback_speed,
            loop=playback_loop,
        )

    if input_device_type == "visionpro":
        from input_devices.visionpro import VisionPro

        return VisionPro(ip=visionpro_ip)

    if input_device_type == "video":
        from input_devices.video_mediapipe import VideoMediaPipe

        return VideoMediaPipe(
            video_path=video_path,
            hand_side=hand_side,
            playback_speed=playback_speed,
            loop=playback_loop,
            video_config=video_config,
            show_video=show_video,
        )

    if input_device_type == "realsense":
        from input_devices.realsense_mediapipe import RealsenseMediaPipe

        return RealsenseMediaPipe(
            hand_side=hand_side,
            video_config=video_config,
            show_video=show_video,
        )

    if input_device_type == "zed":
        from input_devices.zed_mediapipe import ZedMediaPipe

        return ZedMediaPipe(
            hand_side=hand_side,
            video_config=video_config,
            show_video=show_video,
        )

    raise ValueError(f"Unknown input device type: {input_device_type}")


class OmniHandMujocoJointDriver:
    """Write OmniHand active-joint retargeting results into MuJoCo qpos."""

    def __init__(self, mujoco, model, data, robot):
        self.mujoco = mujoco
        self.model = model
        self.data = data
        self.robot = robot
        self.joint_qpos_addr = self._build_joint_qpos_addr()

        active_found = [
            name for name in self.robot.active_joint_names if name in self.joint_qpos_addr
        ]
        if not active_found:
            raise RuntimeError(
                "No OmniHand active joints were found in the MuJoCo model. "
                "Check that the OmniHand URDF loaded successfully."
            )

    def _build_joint_qpos_addr(self) -> dict[str, int]:
        joint_qpos_addr = {}
        for joint_name in self.robot.full_joint_names:
            joint_id = self.mujoco.mj_name2id(
                self.model,
                self.mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )
            if joint_id >= 0:
                joint_qpos_addr[joint_name] = int(self.model.jnt_qposadr[joint_id])
        return joint_qpos_addr

    def set_active_q(self, q_active: np.ndarray):
        q_full = self.robot.active_to_full(q_active)
        for joint_name, value in zip(self.robot.full_joint_names, q_full):
            qpos_addr = self.joint_qpos_addr.get(joint_name)
            if qpos_addr is not None:
                self.data.qpos[qpos_addr] = value


def _set_camera(viewer, hand_side: str):
    viewer.cam.azimuth = 155 if hand_side == "right" else 205
    viewer.cam.elevation = -25
    viewer.cam.distance = 0.38
    viewer.cam.lookat[:] = [0.0, 0.0, 0.04]


def _cleanup_input_device(input_device):
    for method_name in ("stop", "cleanup", "close"):
        method = getattr(input_device, method_name, None)
        if callable(method):
            try:
                method()
            except Exception:
                pass


def run_omnihand_mujoco(
    hand_side: str,
    config_path: str,
    input_device_type: str,
    replay_path: str,
    video_path: str,
    visionpro_ip: str,
    playback_speed: float,
    playback_loop: bool,
    show_video: bool,
    max_frames: int,
    print_every: int,
):
    hand_side = hand_side.lower()
    if hand_side not in {"left", "right"}:
        raise ValueError(f"hand_side must be left or right, got {hand_side}")

    mujoco = _import_mujoco()

    config_file = _resolve_example_path(config_path)
    if config_file is None or not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    retargeter = _create_retargeter(config_file, hand_side)
    robot = retargeter.optimizer.robot
    if not hasattr(robot, "active_to_full"):
        raise TypeError("This viewer requires an OmniHandRobotWrapper-compatible robot.")

    model = mujoco.MjModel.from_xml_path(robot.urdf_path)
    data = mujoco.MjData(model)
    joint_driver = OmniHandMujocoJointDriver(mujoco, model, data, robot)
    safety = OmniHandSafetyFilter(robot.joint_limits)

    q_mid = robot.joint_limits.mean(axis=1)
    joint_driver.set_active_q(q_mid)
    mujoco.mj_forward(model, data)

    input_device = _create_input_device(
        input_device_type=input_device_type,
        hand_side=hand_side,
        config_file=config_file,
        replay_path=replay_path,
        video_path=video_path,
        visionpro_ip=visionpro_ip,
        playback_speed=playback_speed,
        playback_loop=playback_loop,
        show_video=show_video,
    )

    viewer = mujoco.viewer.launch_passive(model, data)
    _set_camera(viewer, hand_side)

    frame_count = 0
    start_time = time.time()
    try:
        print("Starting OmniHand MuJoCo visualization...")
        print(f"  Config: {config_file}")
        print(f"  URDF: {robot.urdf_path}")
        print(f"  Hand: {hand_side}")
        print(f"  Input: {input_device_type}")
        print("=" * 50)

        while viewer.is_running() and (max_frames <= 0 or frame_count < max_frames):
            fingers_data = input_device.get_fingers_data()
            fingers_pose = fingers_data[f"{hand_side}_fingers"]

            if np.allclose(fingers_pose, 0):
                if hasattr(input_device, "is_finished") and input_device.is_finished():
                    break
                time.sleep(0.01)
                continue

            q_raw, verbose = retargeter.retarget_verbose(fingers_pose)
            q_cmd = safety.next(q_raw)
            joint_driver.set_active_q(q_cmd)

            mujoco.mj_forward(model, data)
            viewer.sync()

            frame_count += 1
            if print_every > 0 and (frame_count == 1 or frame_count % print_every == 0):
                elapsed = max(time.time() - start_time, 1e-6)
                fps = frame_count / elapsed
                print(
                    f"frame={frame_count:05d} fps={fps:5.1f} "
                    f"cost={verbose['cost']:.4f} q_active={np.round(q_cmd, 4).tolist()}"
                )

            time.sleep(max(float(model.opt.timestep), 0.001))

    except KeyboardInterrupt:
        print("\nStopping OmniHand MuJoCo visualization...")
    finally:
        viewer.close()
        _cleanup_input_device(input_device)


def main():
    parser = argparse.ArgumentParser(
        description="OmniHand 2025 retargeting with MuJoCo visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mjpython teleop_omnihand_mujoco.py --hand right --play data/avp1.pkl
  mjpython teleop_omnihand_mujoco.py --hand left --frames 300
  mjpython teleop_omnihand_mujoco.py --hand right --video data/right.mp4 --show-video
  mjpython teleop_omnihand_mujoco.py --hand right --realsense
  mjpython teleop_omnihand_mujoco.py --hand right --input visionpro --ip <your-vision-pro-ip>
        """,
    )
    parser.add_argument("--hand", choices=["left", "right"], default="right")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--input",
        choices=["visionpro", "mediapipe_replay", "video", "realsense", "zed"],
        default=None,
    )
    parser.add_argument("--play", type=str, default=None, metavar="FILE")
    parser.add_argument("--video", type=str, default=None, metavar="FILE")
    parser.add_argument("--realsense", action="store_true")
    parser.add_argument("--zed", action="store_true")
    parser.add_argument("--show-video", action="store_true")
    parser.add_argument("--ip", type=str, default="192.168.50.127")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--no-loop", action="store_true")
    parser.add_argument("--frames", type=int, default=0, help="0 means run until viewer closes")
    parser.add_argument("--print-every", type=int, default=30)
    args = parser.parse_args()

    version_error = _python_version_error()
    if version_error:
        print(f"[blocked] {version_error}", file=sys.stderr)
        return 2

    input_device_type = args.input
    replay_path = ""
    video_path = ""

    if args.zed:
        input_device_type = "zed"
    elif args.realsense:
        input_device_type = "realsense"
    elif args.video:
        input_device_type = "video"
        video_path = args.video
    elif args.play:
        input_device_type = "mediapipe_replay"
        replay_path = args.play

    if input_device_type is None:
        input_device_type = "mediapipe_replay"
        replay_path = "data/avp1.pkl"

    if input_device_type == "mediapipe_replay" and not replay_path:
        replay_path = "data/avp1.pkl"
    if input_device_type == "video" and not video_path:
        parser.error("--video FILE is required for video input")

    config_path = args.config or _default_config_for_hand(args.hand)

    try:
        run_omnihand_mujoco(
            hand_side=args.hand,
            config_path=config_path,
            input_device_type=input_device_type,
            replay_path=replay_path,
            video_path=video_path,
            visionpro_ip=args.ip,
            playback_speed=args.speed,
            playback_loop=not args.no_loop,
            show_video=args.show_video,
            max_frames=args.frames,
            print_every=args.print_every,
        )
    except (FileNotFoundError, ImportError, RuntimeError, TypeError, ValueError) as exc:
        print(f"[blocked] {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
