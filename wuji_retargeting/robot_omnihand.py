"""OmniHand 2025 kinematics wrapper.

The OmniHand hardware exposes 10 actively controlled joints, while the URDF
contains additional passive/mimic joints. Retargeting should optimize the 10
active joints and derive passive joints through the same coupling functions
used by the OmniHand SDK.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import numpy.typing as npt
import pinocchio as pin


_PACKAGE_ROOT = Path(__file__).resolve().parent


def _poly_value(x: float, coeffs: List[float]) -> float:
    value = 0.0
    power = 1.0
    for coeff in coeffs:
        value += coeff * power
        power *= x
    return value


def _poly_grad(x: float, coeffs: List[float]) -> float:
    value = 0.0
    power = 1.0
    for i in range(1, len(coeffs)):
        value += i * coeffs[i] * power
        power *= x
    return value


class OmniHandRobotWrapper:
    """Pinocchio wrapper for OmniHand active-joint retargeting.

    Optimizer variables are the 10 active OmniHand joints. Passive joints are
    filled from SDK polynomial coupling before FK/Jacobian evaluation.
    """

    RIGHT_ACTIVE_JOINT_NAMES = [
        "R_thumb_roll_joint",
        "R_thumb_abad_joint",
        "R_thumb_mcp_joint",
        "R_index_abad_joint",
        "R_index_pip_joint",
        "R_middle_pip_joint",
        "R_ring_abad_joint",
        "R_ring_pip_joint",
        "R_pinky_abad_joint",
        "R_pinky_pip_joint",
    ]

    RIGHT_FULL_JOINT_NAMES = [
        "R_thumb_roll_joint",
        "R_thumb_abad_joint",
        "R_thumb_mcp_joint",
        "R_thumb_pip_joint",
        "R_thumb_dip_joint",
        "R_index_abad_joint",
        "R_index_pip_joint",
        "R_index_dip_joint",
        "R_middle_pip_joint",
        "R_middle_dip_joint",
        "R_ring_abad_joint",
        "R_ring_pip_joint",
        "R_ring_dip_joint",
        "R_pinky_abad_joint",
        "R_pinky_pip_joint",
        "R_pinky_dip_joint",
    ]

    RIGHT_ACTIVE_LIMITS = np.array(
        [
            [-0.03, 1.12],
            [-1.64, 0.05],
            [0.0, 0.8416],
            [-0.16, 0.0],
            [0.0, 1.48],
            [0.0, 1.48],
            [0.0, 0.17],
            [0.0, 1.48],
            [0.0, 0.19],
            [0.0, 1.48],
        ],
        dtype=np.float64,
    )

    LEFT_POS_DIRECTION = np.array(
        [-1, -1, -1, -1, 1, 1, -1, 1, -1, 1],
        dtype=np.float64,
    )

    FINGER_PIP_TO_DIP_POLY = [0.0, 2.192, -1.425, 0.747, -0.167]
    THUMB_MCP_TO_PIP_POLY = [0.0, 1.33]
    RIGHT_THUMB_MCP_TO_DIP_POLY = [0.0, 1.846, -0.853, 0.280]

    def __init__(
        self,
        urdf_path: Optional[str] = None,
        hand_side: str = "right",
    ):
        self.hand_side = hand_side.lower()
        if self.hand_side not in {"right", "left"}:
            raise ValueError(f"hand_side must be right or left, got {hand_side}")

        if urdf_path is None:
            urdf_path = str(
                _PACKAGE_ROOT
                / "omnihand_description"
                / "urdf"
                / f"omnihand_{self.hand_side}.urdf"
            )

        self.urdf_path = str(Path(urdf_path).expanduser().resolve())
        self.model: pin.Model = pin.buildModelFromUrdf(self.urdf_path)
        self.data: pin.Data = self.model.createData()

        if self.model.nv != self.model.nq:
            raise NotImplementedError("Cannot handle OmniHand model with special joint.")

        self.prefix = "R" if self.hand_side == "right" else "L"
        self.num_opt_joints = 10
        self.num_joints = self.num_opt_joints

        self.active_joint_names = self._side_names(self.RIGHT_ACTIVE_JOINT_NAMES)
        self.full_joint_names = self._side_names(self.RIGHT_FULL_JOINT_NAMES)
        self._joint_q_index = self._build_joint_q_index()
        self._last_model_q = np.zeros(self.model.nq, dtype=np.float64)

        self._tip_aliases = self._build_link_aliases()

    @property
    def joint_limits(self) -> np.ndarray:
        """Return active joint limits as (lower, upper) pairs."""
        if self.hand_side == "right":
            return self.RIGHT_ACTIVE_LIMITS.copy()

        limits = self.RIGHT_ACTIVE_LIMITS.copy()
        for i, direction in enumerate(self.LEFT_POS_DIRECTION):
            if direction < 0:
                lower, upper = limits[i]
                limits[i] = [-upper, -lower]
        return limits

    @property
    def dof_joint_names(self) -> List[str]:
        return self.active_joint_names.copy()

    def _side_names(self, right_names: List[str]) -> List[str]:
        if self.hand_side == "right":
            return right_names.copy()
        return [name.replace("R_", "L_", 1) for name in right_names]

    def _build_joint_q_index(self) -> Dict[str, int]:
        joint_q_index: Dict[str, int] = {}
        for joint_id, name in enumerate(self.model.names):
            if joint_id == 0 or self.model.nqs[joint_id] == 0:
                continue
            if self.model.nqs[joint_id] != 1:
                raise NotImplementedError(
                    f"Only single-DoF OmniHand joints are supported, got {name}"
                )
            joint_q_index[name] = int(self.model.idx_qs[joint_id])
        return joint_q_index

    def _build_link_aliases(self) -> Dict[str, str]:
        names = {
            "palm_link": f"{self.prefix}_palm",
            "finger1_link3": f"{self.prefix}_thumb_pip",
            "finger1_link4": f"{self.prefix}_thumb_dip",
            "finger1_tip_link": f"{self.prefix}_thumb_tip",
            "finger2_link3": f"{self.prefix}_index_pip",
            "finger2_link4": f"{self.prefix}_index_dip",
            "finger2_tip_link": f"{self.prefix}_index_tip",
            "finger3_link3": f"{self.prefix}_middle_pip",
            "finger3_link4": f"{self.prefix}_middle_dip",
            "finger3_tip_link": f"{self.prefix}_middle_tip",
            "finger4_link3": f"{self.prefix}_ring_pip",
            "finger4_link4": f"{self.prefix}_ring_dip",
            "finger4_tip_link": f"{self.prefix}_ring_tip",
            "finger5_link3": f"{self.prefix}_pinky_pip",
            "finger5_link4": f"{self.prefix}_pinky_dip",
            "finger5_tip_link": f"{self.prefix}_pinky_tip",
        }
        return names

    def get_link_index(self, name: str) -> int:
        """Get frame index by exact name or Wuji-style alias."""
        candidates = [name]
        if name in self._tip_aliases:
            candidates.append(self._tip_aliases[name])
        if not name.startswith(f"{self.prefix}_"):
            candidates.append(f"{self.prefix}_{name}")

        for candidate in candidates:
            idx = self.model.getFrameId(candidate, pin.BODY)
            if idx < self.model.nframes:
                return idx

        available = [self.model.frames[i].name for i in range(self.model.nframes)]
        raise RuntimeError(f"Frame '{name}' not found. Available: {available}")

    def active_to_full(self, q_active: npt.NDArray) -> np.ndarray:
        """Map 10 active OmniHand joints to 16 active+passive joints."""
        q_active = np.asarray(q_active, dtype=np.float64)
        if q_active.shape != (self.num_opt_joints,):
            raise ValueError(f"Expected q_active shape (10,), got {q_active.shape}")

        q_full = np.zeros(16, dtype=np.float64)

        thumb_mcp_to_dip = self.RIGHT_THUMB_MCP_TO_DIP_POLY.copy()
        if self.hand_side == "left":
            thumb_mcp_to_dip[2] = -thumb_mcp_to_dip[2]

        q_full[0] = q_active[0]
        q_full[1] = q_active[1]
        q_full[2] = q_active[2]
        q_full[3] = _poly_value(q_active[2], self.THUMB_MCP_TO_PIP_POLY)
        q_full[4] = _poly_value(q_active[2], thumb_mcp_to_dip)

        q_full[5] = q_active[3]
        q_full[6] = q_active[4]
        q_full[7] = _poly_value(q_active[4], self.FINGER_PIP_TO_DIP_POLY)

        q_full[8] = q_active[5]
        q_full[9] = _poly_value(q_active[5], self.FINGER_PIP_TO_DIP_POLY)

        q_full[10] = q_active[6]
        q_full[11] = q_active[7]
        q_full[12] = _poly_value(q_active[7], self.FINGER_PIP_TO_DIP_POLY)

        q_full[13] = q_active[8]
        q_full[14] = q_active[9]
        q_full[15] = _poly_value(q_active[9], self.FINGER_PIP_TO_DIP_POLY)

        return q_full

    def active_to_model_q(self, q_active: npt.NDArray) -> np.ndarray:
        """Map 10 active joints to the Pinocchio model q vector."""
        q_full = self.active_to_full(q_active)
        q_model = np.zeros(self.model.nq, dtype=np.float64)

        for joint_name, value in zip(self.full_joint_names, q_full):
            q_idx = self._joint_q_index.get(joint_name)
            if q_idx is not None:
                q_model[q_idx] = value

        return q_model

    def _dqmodel_dqactive(self, q_active: npt.NDArray) -> np.ndarray:
        """Compute chain-rule map from active joints to model q."""
        q_active = np.asarray(q_active, dtype=np.float64)
        mat = np.zeros((self.model.nv, self.num_opt_joints), dtype=np.float64)

        def set_entry(joint_name: str, active_idx: int, value: float):
            q_idx = self._joint_q_index.get(joint_name)
            if q_idx is not None:
                mat[q_idx, active_idx] = value

        thumb_mcp_to_dip = self.RIGHT_THUMB_MCP_TO_DIP_POLY.copy()
        if self.hand_side == "left":
            thumb_mcp_to_dip[2] = -thumb_mcp_to_dip[2]

        for active_idx, joint_name in enumerate(self.active_joint_names):
            set_entry(joint_name, active_idx, 1.0)

        set_entry(
            self.full_joint_names[3],
            2,
            _poly_grad(q_active[2], self.THUMB_MCP_TO_PIP_POLY),
        )
        set_entry(
            self.full_joint_names[4],
            2,
            _poly_grad(q_active[2], thumb_mcp_to_dip),
        )
        for full_idx, active_idx in [(7, 4), (9, 5), (12, 7), (15, 9)]:
            set_entry(
                self.full_joint_names[full_idx],
                active_idx,
                _poly_grad(q_active[active_idx], self.FINGER_PIP_TO_DIP_POLY),
            )

        return mat

    def compute_forward_kinematics(self, qpos: npt.NDArray):
        """Compute FK for the active-joint state."""
        self._last_model_q = self.active_to_model_q(qpos)
        pin.forwardKinematics(self.model, self.data, self._last_model_q)

    def get_link_pose(self, link_id: int) -> npt.NDArray:
        pose: pin.SE3 = pin.updateFramePlacement(self.model, self.data, link_id)
        return pose.homogeneous

    def compute_all_jacobians_batch(
        self,
        qpos: npt.NDArray,
        link_indices: List[int],
    ) -> npt.NDArray:
        """Batch compute position Jacobians with respect to active joints."""
        q_model = self.active_to_model_q(qpos)
        dqmodel_dqactive = self._dqmodel_dqactive(qpos)

        pin.computeJointJacobians(self.model, self.data, q_model)
        pin.updateFramePlacements(self.model, self.data)

        jacobians = []
        for idx in link_indices:
            J_local = pin.getFrameJacobian(self.model, self.data, idx, pin.LOCAL)
            R = self.data.oMf[idx].rotation
            J_world_pos_model = R @ J_local[:3, :]
            jacobians.append(J_world_pos_model @ dqmodel_dqactive)

        return np.stack(jacobians, axis=0)

    def compute_fk_batch(
        self,
        qpos: npt.NDArray,
        link_indices: List[int],
    ) -> npt.NDArray:
        """Batch compute FK positions for multiple links."""
        q_model = self.active_to_model_q(qpos)
        pin.forwardKinematics(self.model, self.data, q_model)
        pin.updateFramePlacements(self.model, self.data)

        positions = []
        for idx in link_indices:
            positions.append(self.data.oMf[idx].translation)

        return np.concatenate(positions)
