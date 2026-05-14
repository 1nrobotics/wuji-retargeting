# OmniHand Retargeting 算法适配计划

本文档面向工程实现，目标是把当前 `wuji-retargeting` 的几何优化 retargeting 管线适配到 OmniHand 2025 灵动款灵巧手。

## 1. 背景与目标

当前 `wuji-retargeting` 的核心流程是：

```text
MediaPipe/VisionPro/RealSense/ZED 输入
  -> (21, 3) 人手关键点
  -> wrist frame 坐标变换
  -> 构造 retargeting 目标向量
  -> Pinocchio FK/Jacobian
  -> NLopt SLSQP 优化
  -> Wuji Hand 20 维 qpos
```

OmniHand 2025 SDK 提供：

```text
10 个 active joints
16 个 total joints = 10 active + 6 passive/mimic joints
左右手 URDF
Python/C++ 硬件控制 API
active joint angle 控制接口
active -> passive 的多项式耦合关系
```

适配目标：

```text
MediaPipe/VisionPro/RealSense/ZED 输入
  -> (21, 3) 人手关键点
  -> OmniHand 10 维 active joint angle
  -> 可用于离线 FK 验证、仿真验证和真机 set_all_active_joint_angles()
```

## 2. 设计原则

1. 不改动现有 Wuji Hand 路径的行为。
2. 优先复用现有输入设备、坐标变换、低通滤波和优化框架。
3. 第一阶段优先实现 `VectorOptimizer`，暂不迁移完整 Adaptive loss。
4. OmniHand 优化变量必须是 10 维 active joints，而不是 16 维 full joints。
5. Passive joints 通过 SDK 中的多项式耦合从 active joints 计算。
6. FK/Jacobian 使用 OmniHand URDF + Pinocchio。
7. 真机控制前必须加入关节限位、速度限制、丢帧保护和急停接口。

## 3. OmniHand 关节定义

### 3.1 Active Joint 顺序

右手 active joint 顺序：

```text
0  R_thumb_roll_joint
1  R_thumb_abad_joint
2  R_thumb_mcp_joint
3  R_index_abad_joint
4  R_index_pip_joint
5  R_middle_pip_joint
6  R_ring_abad_joint
7  R_ring_pip_joint
8  R_pinky_abad_joint
9  R_pinky_pip_joint
```

左手 active joint 顺序：

```text
0  L_thumb_roll_joint
1  L_thumb_abad_joint
2  L_thumb_mcp_joint
3  L_index_abad_joint
4  L_index_pip_joint
5  L_middle_pip_joint
6  L_ring_abad_joint
7  L_ring_pip_joint
8  L_pinky_abad_joint
9  L_pinky_pip_joint
```

### 3.2 Full Joint 顺序

右手 full joint 顺序建议对齐 SDK `OmnihandJoint`：

```text
0  R_thumb_roll_joint
1  R_thumb_abad_joint
2  R_thumb_mcp_joint
3  R_thumb_pip_joint
4  R_thumb_dip_joint
5  R_index_abad_joint
6  R_index_pip_joint
7  R_index_dip_joint
8  R_middle_pip_joint
9  R_middle_dip_joint
10 R_ring_abad_joint
11 R_ring_pip_joint
12 R_ring_dip_joint
13 R_pinky_abad_joint
14 R_pinky_pip_joint
15 R_pinky_dip_joint
```

左手同理，将 `R_` 替换为 `L_`。

### 3.3 Active 到 Full 的耦合

SDK 中 `kinematics_solver.cc` 给出了 passive joint 的多项式关系：

```text
thumb_pip = poly_thumb_mcp_to_pip(thumb_mcp)
thumb_dip = poly_thumb_mcp_to_dip(thumb_mcp)
finger_dip = poly_finger_pip_to_dip(finger_pip)
```

右手默认系数：

```text
finger_pip2dip_poly = [0.0, 2.192, -1.425, 0.747, -0.167]
thumb_mcp2pip_poly = [0.0, 1.33]
thumb_mcp2dip_poly = [0.0, 1.846, -0.853, 0.280]
```

左手 SDK 里会对部分方向和拇指 DIP 多项式做符号调整。实现时应复刻 SDK 逻辑，而不是只使用 URDF mimic multiplier。

## 4. 推荐新增文件

建议新增如下文件：

```text
wuji_retargeting/robots/__init__.py
wuji_retargeting/robots/base.py
wuji_retargeting/robots/wuji.py
wuji_retargeting/robots/omnihand.py

wuji_retargeting/opt/omnihand_vector.py        # 可选，如果不想改 VectorOptimizer
example/config/omnihand/vector_omnihand_right.yaml
example/config/omnihand/vector_omnihand_left.yaml
example/teleop_omnihand.py
```

如果希望最小改动，也可以先不做 `robots/` 抽象，只新增：

```text
wuji_retargeting/robot_omnihand.py
wuji_retargeting/opt/omnihand_vector.py
example/config/omnihand/vector_omnihand_right.yaml
example/teleop_omnihand.py
```

长期建议做机器人抽象层，避免 `BaseOptimizer` 继续写死 Wuji URDF 和 link 名称。

## 5. 机器人抽象接口

建议定义统一 RobotWrapper 接口：

```python
class RobotKinematicsBase:
    num_opt_joints: int
    joint_limits: np.ndarray  # shape = (num_opt_joints, 2)

    def get_link_index(self, name: str) -> int:
        ...

    def compute_fk_batch(
        self,
        q_opt: np.ndarray,
        link_indices: list[int],
    ) -> np.ndarray:
        """Return flattened positions, shape = (num_links * 3,)."""

    def compute_all_jacobians_batch(
        self,
        q_opt: np.ndarray,
        link_indices: list[int],
    ) -> np.ndarray:
        """Return position Jacobians, shape = (num_links, 3, num_opt_joints)."""
```

Wuji 的 `num_opt_joints = 20`。

OmniHand 的 `num_opt_joints = 10`。

## 6. OmniHandRobotWrapper 设计

### 6.1 初始化

输入：

```python
OmniHandRobotWrapper(
    urdf_path: str,
    hand_side: Literal["left", "right"],
)
```

初始化时：

1. 使用 Pinocchio 加载 OmniHand URDF。
2. 建立 active joint name -> Pinocchio q index 映射。
3. 建立 full joint name -> Pinocchio q index 映射。
4. 构造 10 维 active joint limits。
5. 保存 active-to-full 多项式耦合。

### 6.2 active_to_full

接口：

```python
def active_to_full(self, q_active: np.ndarray) -> np.ndarray:
    """Map 10 active joints to 16 full joints."""
```

右手逻辑：

```python
q_full[thumb_roll] = q_active[thumb_roll]
q_full[thumb_abad] = q_active[thumb_abad]
q_full[thumb_mcp] = q_active[thumb_mcp]
q_full[thumb_pip] = poly(thumb_mcp, thumb_mcp2pip_poly)
q_full[thumb_dip] = poly(thumb_mcp, thumb_mcp2dip_poly)

q_full[index_abad] = q_active[index_abad]
q_full[index_pip] = q_active[index_pip]
q_full[index_dip] = poly(index_pip, finger_pip2dip_poly)

q_full[middle_pip] = q_active[middle_pip]
q_full[middle_dip] = poly(middle_pip, finger_pip2dip_poly)

q_full[ring_abad] = q_active[ring_abad]
q_full[ring_pip] = q_active[ring_pip]
q_full[ring_dip] = poly(ring_pip, finger_pip2dip_poly)

q_full[pinky_abad] = q_active[pinky_abad]
q_full[pinky_pip] = q_active[pinky_pip]
q_full[pinky_dip] = poly(pinky_pip, finger_pip2dip_poly)
```

注意：Pinocchio 读入 URDF 时是否展开 mimic joint 需要实测。如果 Pinocchio 模型 `nq` 不是 16，而是只包含非 mimic joints，则 wrapper 要按实际 `model.names` 做映射，不能硬编码写入不存在的 mimic q index。

### 6.3 chain rule Jacobian

如果 Pinocchio FK 使用 16 维 full joints，则需要：

```text
J_active = J_full @ d q_full / d q_active
```

其中：

```python
J_full.shape = (num_links, 3, 16)
dqfull_dqactive.shape = (16, 10)
J_active.shape = (num_links, 3, 10)
```

多项式求导：

```python
def poly_val(x, coeffs):
    y = 0.0
    power = 1.0
    for c in coeffs:
        y += c * power
        power *= x
    return y

def poly_grad(x, coeffs):
    dy = 0.0
    power = 1.0
    for i in range(1, len(coeffs)):
        dy += i * coeffs[i] * power
        power *= x
    return dy
```

如果 Pinocchio 模型只包含 active joints，并通过 URDF mimic 自动处理 passive links，则不需要手动 chain rule。但要写单元测试确认 `R_index_tip` 位置会随 `R_index_pip_joint` 正确运动。

## 7. OmniHand VectorOptimizer 设计

第一阶段建议不要直接迁移 AdaptiveOptimizerAnalytical，而是做 OmniHand 版 VectorOptimizer。

优化变量：

```python
q_active: np.ndarray  # shape = (10,)
```

目标向量建议：

```yaml
key_vectors:
  # thumb
  - {origin: R_palm, task: R_thumb_pip, origin_kp: 0, task_kp: 2, scale: 1.0}
  - {origin: R_palm, task: R_thumb_dip, origin_kp: 0, task_kp: 3, scale: 1.0}
  - {origin: R_palm, task: R_thumb_tip, origin_kp: 0, task_kp: 4, scale: 1.0}

  # index
  - {origin: R_palm, task: R_index_pip, origin_kp: 0, task_kp: 6, scale: 1.0}
  - {origin: R_palm, task: R_index_dip, origin_kp: 0, task_kp: 7, scale: 1.0}
  - {origin: R_palm, task: R_index_tip, origin_kp: 0, task_kp: 8, scale: 1.0}

  # middle
  - {origin: R_palm, task: R_middle_pip, origin_kp: 0, task_kp: 10, scale: 1.0}
  - {origin: R_palm, task: R_middle_dip, origin_kp: 0, task_kp: 11, scale: 1.0}
  - {origin: R_palm, task: R_middle_tip, origin_kp: 0, task_kp: 12, scale: 1.0}

  # ring
  - {origin: R_palm, task: R_ring_pip, origin_kp: 0, task_kp: 14, scale: 1.0}
  - {origin: R_palm, task: R_ring_dip, origin_kp: 0, task_kp: 15, scale: 1.0}
  - {origin: R_palm, task: R_ring_tip, origin_kp: 0, task_kp: 16, scale: 1.0}

  # pinky
  - {origin: R_palm, task: R_pinky_pip, origin_kp: 0, task_kp: 18, scale: 1.0}
  - {origin: R_palm, task: R_pinky_dip, origin_kp: 0, task_kp: 19, scale: 1.0}
  - {origin: R_palm, task: R_pinky_tip, origin_kp: 0, task_kp: 20, scale: 1.0}
```

Loss：

```text
L(q_active) =
    mean_i Huber(||robot_vec_i(q_active) - target_vec_i||)
  + norm_delta * ||q_active - q_prev||^2
```

输出：

```python
q_active: np.ndarray  # shape = (10,)
```

## 8. 配置文件示例

新增：

```text
example/config/omnihand/vector_omnihand_right.yaml
```

示例结构：

```yaml
robot:
  type: "OmniHand"
  hand_side: "right"
  urdf_path: "/absolute/or/relative/path/to/omnihand_right.urdf"
  origin_link: "R_palm"
  active_joint_names:
    - R_thumb_roll_joint
    - R_thumb_abad_joint
    - R_thumb_mcp_joint
    - R_index_abad_joint
    - R_index_pip_joint
    - R_middle_pip_joint
    - R_ring_abad_joint
    - R_ring_pip_joint
    - R_pinky_abad_joint
    - R_pinky_pip_joint

optimizer:
  type: "OmniHandVectorOptimizer"

retarget:
  huber_delta: 2.0
  norm_delta: 0.04
  lp_alpha: 0.2
  mediapipe_rotation:
    x: 0.0
    y: 0.0
    z: 0.0
  key_vectors:
    # fill as above
```

左手配置把 `R_` 换成 `L_`，并设置 `hand_side: left`。

## 9. 真机控制 Adapter

新增 `example/teleop_omnihand.py`。

核心逻辑：

```python
from omnihand_2025 import AgibotHandO10, EHandType

hand = AgibotHandO10.create_hand(hand_type=EHandType.RIGHT)

retargeter = Retargeter.from_yaml(config_path, hand_side="right")

while True:
    fingers_data = input_device.get_fingers_data()
    keypoints = fingers_data["right_fingers"]
    q_active = retargeter.retarget(keypoints)
    q_active = safety_filter(q_active)
    hand.set_all_active_joint_angles(q_active.tolist())
```

### 9.1 SafetyFilter

必须实现：

```python
class OmniHandSafetyFilter:
    def __init__(self, joint_limits, velocity_limits, max_dt=0.05):
        ...

    def next(self, q_target, timestamp=None):
        q = clip_to_joint_limits(q_target)
        q = clip_velocity(q, q_prev, dt)
        q = low_pass(q)
        return q
```

原因：OmniHand 文档里的速度限制较低：

```text
abad/roll: 0.164 rad/s
pip/mcp:  0.308 rad/s
```

真机发送前不能只依赖 optimizer 的输出。

### 9.2 dry-run 模式

当前 `example/teleop_omnihand.py` 默认运行在 dry-run 模式。dry-run 的目的不是控制真机，而是在不连接 OmniHand 硬件、不 import OmniHand SDK 的情况下，验证完整 retargeting 数值链路：

```text
MediaPipe replay 输入
  -> Retargeter
  -> VectorOptimizer
  -> OmniHandRobotWrapper FK/Jacobian
  -> 10 维 q_active
  -> OmniHandSafetyFilter
  -> 终端打印 q_active / cost / FPS
```

默认输入数据是仓库中的回放文件：

```text
example/data/avp1.pkl
```

命令：

```bash
python3.10 example/teleop_omnihand.py --hand right --frames 30
```

等价地，也可以显式写出 dry-run：

```bash
python3.10 example/teleop_omnihand.py --hand right --frames 30 --dry-run
```

内部使用 `MediaPipeReplay(record_path="data/avp1.pkl")` 读取预录数据。每一帧输入格式是：

```python
{
    "left_fingers": np.ndarray,   # shape = (21, 3)
    "right_fingers": np.ndarray,  # shape = (21, 3)
}
```

如果指定右手，则读取 `right_fingers`；如果指定左手，则读取 `left_fingers`。输出打印类似：

```text
frame=00030 fps= 28.7 cost=0.1234 q_active=[...10 values...]
```

其中 `q_active` 是可以发送给 OmniHand 的 10 维主动关节角，但 dry-run 下不会调用：

```python
hand.set_all_active_joint_angles(q_cmd.tolist())
```

如果要换输入文件：

```bash
python3.10 example/teleop_omnihand.py --hand right --play data/your_replay.pkl --frames 300
```

### 9.3 dry-run 下的 OmniHand 可视化

当前已新增 `example/teleop_omnihand_mujoco.py`，用于在 dry-run / replay / video / camera 输入下显示 OmniHand 的 MuJoCo 可视化。现有 `wuji_retargeting.viz.TuningViewer` 基于 Wuji Hand 的 MuJoCo MJCF、Wuji link 命名和 20 维 qpos，不能直接拿来显示 OmniHand，因此 OmniHand 使用独立入口。

运行示例：

```bash
cd example
mjpython teleop_omnihand_mujoco.py --hand right --play data/avp1.pkl
```

默认输入仍然是 `example/data/avp1.pkl`：

```bash
cd example
mjpython teleop_omnihand_mujoco.py --hand right
```

内部流程：

```text
MediaPipe 21 点
  -> Retargeter / VectorOptimizer
  -> q_active, shape = (10,)
  -> OmniHandSafetyFilter
  -> OmniHandRobotWrapper.active_to_full(q_active), shape = (16,)
  -> 按 joint name 写入 MuJoCo data.qpos
  -> mj_forward()
  -> viewer.sync()
```

OmniHand 可视化分两层：

```text
第一层：URDF mesh 可视化，当前已实现
  输入 MediaPipe 21 点
  优化得到 q_active
  OmniHandRobotWrapper.active_to_full(q_active)
  MuJoCo 加载 OmniHand URDF 和 STL mesh
  直接写 data.qpos 显示手部姿态

第二层：三层 skeleton tuning viewer，后续可做
  orange: MediaPipe input skeleton
  cyan:   scaled target vectors
  white:  OmniHand FK skeleton
```

当前实现使用 SDK URDF + STL mesh，而不是 MuJoCo actuator。原因是 OmniHand SDK 只提供 URDF，没有现成 MJCF actuator 定义；因此 viewer 只做可视化，不做动力学控制。写入逻辑复用以下接口：

```python
robot = retargeter.optimizer.robot
q_active, verbose = retargeter.retarget_verbose(fingers_pose)
q_full = robot.active_to_full(q_active)
```

支持输入：

```bash
mjpython teleop_omnihand_mujoco.py --hand right --play data/avp1.pkl
mjpython teleop_omnihand_mujoco.py --hand right --video data/right.mp4 --show-video
mjpython teleop_omnihand_mujoco.py --hand right --realsense
mjpython teleop_omnihand_mujoco.py --hand right --zed
mjpython teleop_omnihand_mujoco.py --hand right --input visionpro --ip <vision-pro-ip>
```

后续如果要做与 Wuji `TuningViewer` 同级的三层 skeleton 调参工具，可以在 `wuji_retargeting.viz` 中新增 OmniHand link mapping，把 `q_active -> FK points` 画成白色 robot skeleton，并支持 YAML hot-reload。

## 10. 验证计划

### 10.1 静态验证

1. 能加载 OmniHand URDF。
2. 能列出 Pinocchio 中的 joint names 和 frame names。
3. active joint limits 与 SDK 文档一致。
4. `active_to_full()` 输出长度正确。
5. left/right 方向与 SDK 逻辑一致。

### 10.2 FK 验证

用 SDK 示例姿态验证：

```python
paper = [0.58, -0.21, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
fist = [0.43, -0.3, 0.66, 0.0, 1.48, 1.48, 0.0, 1.48, 0.0, 1.48]
ok = [-0.03, -1.51, 0.7, -0.16, 0.85, 0.21, 0.07, 0.153, 0.107, 0.1]
```

检查：

```text
paper: 手指基本伸展
fist: 手指明显弯曲
ok: 拇指和食指靠近
```

### 10.3 Jacobian 验证

对若干随机 `q_active` 做有限差分检查：

```text
J_analytical = wrapper.compute_all_jacobians_batch(q)
J_fd = finite_difference_fk(q)
max_abs_error < 1e-4 或相对误差可接受
```

### 10.4 Retargeting 离线验证

用已有 `example/data/avp1.pkl` 或视频输入：

```text
input keypoints
  -> q_active
  -> FK tip positions
```

可视化三层：

```text
raw MediaPipe skeleton
target vectors
OmniHand FK skeleton
```

### 10.5 真机 dry-run

在不连接硬件或不使能电机时：

```text
打印 q_active
检查范围
检查速度
检查 NaN/Inf
检查输入丢帧行为
```

### 10.6 真机低速验证

第一轮真机：

```text
只发送低频命令，例如 5-10 Hz
只跑回放数据
速度限制设为文档限制的 20%-30%
确认无异常后再逐步提高频率
```

## 11. 里程碑

### Milestone 1: Offline Kinematics

交付：

```text
OmniHandRobotWrapper
active_to_full()
FK batch
Jacobian batch
基础单元测试
```

验收：

```text
能加载 URDF
能输出 5 个 tip link 的位置
Jacobian finite-difference 测试通过
```

### Milestone 2: Offline Retargeting

交付：

```text
OmniHandVectorOptimizer
right/left YAML config
离线 pkl 输入 -> q_active 输出
调试打印 cost/FPS
```

验收：

```text
q_active shape = (10,)
q_active 在限位内
回放数据能连续运行
loss 收敛且无 NaN
```

### Milestone 3: Visualization

交付：

```text
OmniHand FK skeleton visualization
输入 skeleton vs robot FK skeleton 对比
基本调参指南
```

验收：

```text
能直观看到五个 fingertip 跟随输入手势
```

### Milestone 4: Hardware Adapter

交付：

```text
example/teleop_omnihand.py
OmniHandSafetyFilter
SDK import fallback/error message
dry-run mode
```

验收：

```text
dry-run 下可以打印稳定的 q_active
连接硬件后可低速发送 set_all_active_joint_angles()
```

## 12. Coding Agent 任务说明

下面这段可以直接交给 coding agent 执行。

```text
你在 wuji-retargeting 仓库中工作。目标是新增 OmniHand 2025 的 retargeting 适配，保留现有 Wuji Hand 行为不变。

参考资料：
- OmniHand SDK zip 位于 /Users/byc/Desktop/forsense/Omnihand-2025-SDK.zip
- 里面有 assets/urdf/omnihand_right.urdf 和 omnihand_left.urdf
- Python SDK 使用 AgibotHandO10.set_all_active_joint_angles(List[float])
- OmniHand 是 10 active joints + 6 passive joints
- active joint 顺序参考 SDK 文档和 src/kinematics_solver/kinematics_solver.h
- passive joints 由 src/kinematics_solver/kinematics_solver.cc 中多项式计算，不要把 16 个 full joints 当成独立优化变量

请分阶段实现：

1. 新增 OmniHandRobotWrapper
   - 支持 right/left URDF
   - 优化变量为 10 维 q_active
   - 实现 active_to_full(q_active)
   - 实现 compute_fk_batch(q_active, link_indices)
   - 实现 compute_all_jacobians_batch(q_active, link_indices)
   - 如果 Pinocchio 模型包含 full joints，则用 chain rule 把 J_full 转为 J_active
   - 如果 Pinocchio 模型只包含 active joints，则确认 mimic joints 在 FK 中正确生效

2. 新增 OmniHandVectorOptimizer
   - 复用现有 VectorOptimizer 的 loss 思路
   - 输入仍然是 transformed MediaPipe keypoints，shape = (21, 3)
   - 输出 q_active，shape = (10,)
   - 使用 OmniHandRobotWrapper 的 FK/Jacobian
   - 支持 YAML 中配置 key_vectors

3. 新增配置文件
   - example/config/omnihand/vector_omnihand_right.yaml
   - example/config/omnihand/vector_omnihand_left.yaml
   - 包含 robot.urdf_path、active_joint_names、origin_link、key_vectors、retarget 参数

4. 新增离线验证脚本或测试
   - 能从 example/data/avp1.pkl 读取数据
   - 能输出连续 q_active
   - 打印 shape、min/max、cost、FPS
   - 不需要连接 OmniHand 真机

5. 新增硬件 adapter 脚本
   - example/teleop_omnihand.py
   - 支持 --dry-run
   - dry-run 模式只打印 q_active，不 import 或连接硬件
   - 非 dry-run 模式使用 omnihand_2025.AgibotHandO10
   - 发送 hand.set_all_active_joint_angles(q_active.tolist())
   - 加 OmniHandSafetyFilter，包括 joint clipping、velocity limiting、low-pass filter、NaN/Inf 检查

6. 测试要求
   - 不要破坏现有 Wuji retargeting
   - 添加针对 active_to_full 的单元测试
   - 添加 FK/Jacobian finite-difference 测试，如果本地缺少 Pinocchio 或 URDF 路径不可用，要给出清晰 skip/error
   - 对新代码运行基本 lint/import 检查

实现注意事项：
- 不要硬编码只支持右手，left/right 都要可配置
- 不要假设输出是 20 维或 reshape(5, 4)
- 不要直接优化 16 个 full joints 后发送给硬件
- 真机发送前必须经过 safety filter
- 保持已有 Retargeter API 尽量兼容，必要时通过 config 中 robot.type 或 optimizer.type 分流
```

## 13. 风险与待确认问题

1. Pinocchio 对 URDF mimic joint 的处理需要实测。
2. SDK 文档中 `get_all_joint_angles()` 返回长度写成 10，可能是文档错误；代码语义应是 active + passive。
3. URDF mesh filename 是开发者本机绝对路径，可能需要重写成 package-relative 或绝对解压路径。
4. Mac 上可以做代码和离线模型开发，但真机 SDK 明确面向 Ubuntu 22.04 x86_64。
5. 左手方向需要严格对照 SDK `left_pos_direction_` 验证。
6. 第一版 key vector scaling 需要调参，不能期待一次达到 Wuji 原手效果。

## 14. 推荐第一步

第一步不要碰真机，也不要做完整 teleop。先实现：

```text
OmniHandRobotWrapper
  -> active_to_full()
  -> FK tip positions
  -> Jacobian finite-difference check
```

只要这一步可靠，后面的 VectorOptimizer 基本就是把 Wuji 现有代码换机器人模型和关节维度。
