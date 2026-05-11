# Wuji Retargeting 算法原理分析

本文档分析 `wuji-retargeting` 代码库中的核心数据格式、retargeting 算法流程、硬件支持范围，并讨论后续基于学习式模型的演进方向。

## 1. 项目定位

`wuji-retargeting` 是面向 Wuji Hand 的手部姿态重定向系统。它的目标是把外部输入设备采集到的人手 3D 关键点，实时转换成 Wuji Hand 的关节目标角。

从代码实现看，本仓库当前采用的是几何/运动学优化方法，而不是端到端神经网络方法。核心流程是：

```text
输入设备数据
  -> MediaPipe 风格 21 个手部 3D 关键点
  -> 坐标系标准化
  -> 构造人手目标向量/方向
  -> 机器人手 URDF 正运动学
  -> 非线性约束优化
  -> 20 维 Wuji Hand 关节角
```

核心代码分布如下：

| 模块 | 作用 |
| --- | --- |
| `wuji_retargeting/retarget.py` | 高层统一接口 `Retargeter` |
| `wuji_retargeting/mediapipe.py` | MediaPipe 关键点坐标变换 |
| `wuji_retargeting/robot.py` | Pinocchio 机器人模型、正运动学和 Jacobian |
| `wuji_retargeting/opt/base.py` | 优化器基类、NLopt 初始化、滤波 |
| `wuji_retargeting/opt/adaptive_analytical.py` | 自适应解析梯度优化器 |
| `wuji_retargeting/opt/vector.py` | 关键向量优化器 |
| `example/input_devices/` | Vision Pro、视频、RealSense、ZED、回放输入 |
| `example/teleop_sim.py` | MuJoCo 仿真控制 |
| `example/teleop_real.py` | Wuji Hand 真机控制 |

当前工作区中 `wuji_retargeting/wuji_hand_description` 和 `example/utils/mujoco-sim` 是 git submodule。正常运行需要初始化它们，否则 URDF 和 MJCF 模型文件不可用。

## 2. 数据输入格式

### 2.1 统一输入格式

所有输入设备最终都被转换为 MediaPipe 风格的手部关键点数组：

```python
raw_keypoints: np.ndarray  # shape = (21, 3)
```

每一行表示一个手部关键点的三维坐标。主要索引如下：

```text
0: wrist

thumb:  1, 2, 3, 4
index:  5, 6, 7, 8
middle: 9, 10, 11, 12
ring:   13, 14, 15, 16
pinky:  17, 18, 19, 20

tips: [4, 8, 12, 16, 20]
PIP:  [2, 6, 10, 14, 18]  # thumb 使用 MCP=2 作为类 PIP 点
DIP:  [3, 7, 11, 15, 19]
```

输入设备接口返回统一字典：

```python
{
    "left_fingers": np.ndarray,   # shape = (21, 3)
    "right_fingers": np.ndarray,  # shape = (21, 3)
}
```

如果某只手没有检测到，通常返回全零数组。

### 2.2 录制文件格式

回放数据是 `.pkl` 文件，内容是帧列表：

```python
[
    {
        "t": float,
        "left_fingers": np.ndarray,   # shape = (21, 3)
        "right_fingers": np.ndarray,  # shape = (21, 3)
    },
    ...
]
```

`t` 是时间戳。若时间戳无效，回放模块会退化为固定 30 FPS 的帧计数模式。

### 2.3 Vision Pro 输入

Vision Pro 原始输入是每只手 25 个关节的 4x4 变换矩阵：

```python
fingers_mat: np.ndarray  # shape = (25, 4, 4)
```

代码通过 `VP_TO_MEDIAPIPE` 映射选取其中 21 个点，并取每个矩阵的平移部分：

```python
mediapipe_pose[mp_idx] = fingers_mat[vp_idx][:3, 3]
```

因此 Vision Pro 数据进入 retargeting 之前也会被转换为 `(21, 3)`。

### 2.4 视频、RealSense、ZED 输入

视频、RealSense 和 ZED 都使用 MediaPipe Hands 做手部关键点检测。流程是：

```text
RGB 图像
  -> MediaPipe Hands landmarks
  -> normalized image coordinates
  -> 像素尺度 3D 点
  -> wrist 居中
  -> 按 wrist 到 middle MCP 距离缩放到近似真实尺度
  -> 可选手指段长修正
  -> (21, 3)
```

单目输入的 `z` 通常不准确，因此配置里提供：

```yaml
video_input:
  z_scale: 2.5
  correct_segments: true
  reference_wrist_to_mid_mcp: 0.09
```

其中 `z_scale` 用于放大 MediaPipe 单目深度，`correct_segments` 用固定人体手指段长修正每根手指的 MCP-PIP、PIP-DIP、DIP-TIP 长度。

## 3. 数据输出格式

### 3.1 标准输出

`Retargeter.retarget()` 的输出是：

```python
qpos: np.ndarray  # shape = (20,)
```

这 20 个值是 Wuji Hand 的关节目标角。语义上对应：

```text
5 fingers x 4 joints = 20 joints
```

真实硬件控制中会 reshape 为：

```python
qpos.reshape(5, 4)
```

然后调用：

```python
handcontroller.set_joint_target_position(qpos.reshape(5, 4))
```

仿真中则直接把 `qpos` 写入 MuJoCo actuator control：

```python
data.ctrl[:] = qpos
```

### 3.2 Verbose 输出

`retarget_verbose()` 返回：

```python
qpos, verbose_dict
```

其中 `verbose_dict` 包含：

```python
{
    "mediapipe_kp": np.ndarray,      # 坐标变换后的关键点，shape = (21, 3)
    "qpos_unfiltered": np.ndarray,   # 未滤波优化输出，shape = (20,)
    "qpos": np.ndarray,              # 低通滤波后的输出，shape = (20,)
    "cost": float,                   # 当前优化代价
    "pinch_alphas": np.ndarray,      # AdaptiveOptimizerAnalytical 专有，shape = (5,)
}
```

这个接口主要服务于调参可视化。

## 4. 坐标系预处理

输入关键点首先经过 `apply_mediapipe_transformations()`。

主要步骤：

1. 所有点减去 wrist 点，使 wrist 成为原点。
2. 使用关键点 `0, 5, 9` 估计手掌局部坐标系。
3. 通过 SVD 估计手掌平面法向。
4. 根据左右手乘以不同的 `OPERATOR2MANO_RIGHT` 或 `OPERATOR2MANO_LEFT` 矩阵。
5. 如果 YAML 配置了 `mediapipe_rotation`，再做额外欧拉角旋转修正。

处理后，关键点仍然是 `(21, 3)`，但已经进入统一的 wrist frame / MANO-like frame。

额外旋转配置示例：

```yaml
retarget:
  mediapipe_rotation:
    x: -5.0
    y: -5.0
    z: -15.0
```

## 5. 优化基础设施

所有优化器继承 `BaseOptimizer`。

初始化时会：

1. 根据 `hand_side` 加载 URDF：

```text
wuji_retargeting/wuji_hand_description/urdf/{left,right}.urdf
```

2. 使用 Pinocchio 构建机器人模型。
3. 读取机器人关节上下限。
4. 初始化 NLopt SLSQP 优化器：

```python
nlopt.opt(nlopt.LD_SLSQP, num_joints)
```

5. 设置优化参数：

```text
maxeval = 50
ftol_abs = 1e-4
lower_bounds = URDF joint lower limits
upper_bounds = URDF joint upper limits
```

6. 保存上一帧优化结果 `last_qpos`，用于 warm start 和时序正则。

如果有上一帧：

```text
init_qpos = last_qpos
```

否则：

```text
init_qpos = mean(joint_limits)
```

通用时序正则为：

```text
norm_delta * ||q - q_prev||^2
```

这能减少输出抖动，也能让相邻帧更容易快速收敛。

## 6. 算法一：AdaptiveOptimizerAnalytical

`AdaptiveOptimizerAnalytical` 是默认优化器。它把两种 retargeting 目标自适应混合：

- `TipDirVec`：适合捏合、精细操作，强调指尖位置和指尖方向。
- `FullHandVec`：适合普通手势，强调整根手指的骨架形状。

### 6.1 Pinch alpha

算法先计算拇指 tip 到其他四指 tip 的距离：

```text
d_i = ||tip_i - thumb_tip||
```

然后根据配置中的 `d1`、`d2` 得到每根手指的混合权重：

```text
alpha_i = clip((d2_i - d_i) / (d2_i - d1_i), 0.0, 0.7)
```

直觉上：

- 距离越近，越像捏合，`alpha` 越大。
- 距离越远，越像普通张手/弯曲，`alpha` 越小。

拇指的 alpha 使用其他四指 alpha 的最大值：

```text
alpha_thumb = max(alpha_index, alpha_middle, alpha_ring, alpha_pinky)
```

### 6.2 目标量构造

每一帧会从 MediaPipe 关键点构造三类目标。

#### 6.2.1 Tip position vectors

从 wrist 到五个 fingertip 的向量：

```text
target_tip_vectors[f] = keypoints[tip_f] - keypoints[wrist]
```

输出单位转换为厘米。

#### 6.2.2 Tip direction vectors

从 DIP 到 fingertip 的单位方向：

```text
target_tip_dirs[f] = normalize(keypoints[tip_f] - keypoints[dip_f])
```

这个目标不关心长度，只关心末端指节方向。

#### 6.2.3 Full hand vectors

每根手指构造：

```text
wrist -> PIP
wrist -> DIP
wrist -> TIP
```

五根手指一共 15 个向量。

配置中的 `segment_scaling` 可以按手指、按段缩放：

```yaml
segment_scaling:
  thumb:  [1.0, 1.0, 1.0]
  index:  [1.0, 1.03, 1.05]
  middle: [1.0, 1.0, 1.0]
  ring:   [1.0, 1.0, 1.0]
  pinky:  [1.05, 1.15, 1.15]
```

### 6.3 机器人 FK 对应量

对于候选关节角 `q`，算法通过 Pinocchio 计算：

```text
palm_link
finger{i}_link3
finger{i}_link4
finger{i}_tip_link
```

然后得到：

```text
robot_tip_vec = tip_link - palm_link
robot_tip_dir = normalize(tip_link - link4)
robot_pip_vec = link3 - palm_link
robot_dip_vec = link4 - palm_link
robot_tip_vec_full = tip_link - palm_link
```

同时批量计算这些 link 对关节角的 Jacobian，用于解析梯度。

### 6.4 Loss 函数

每根手指的 TipDirVec loss：

```text
L_tip_dir_vec_i =
    w_pos * Huber(||robot_tip_vec_i - target_tip_vec_i||)
  + w_dir * Huber(||robot_tip_dir_i - target_tip_dir_i||)
```

FullHandVec loss：

```text
L_full_hand_i =
    w_full_hand / 3 * (
        Huber(||robot_pip_vec_i - target_pip_i||)
      + Huber(||robot_dip_vec_i - target_dip_i||)
      + Huber(||robot_tip_vec_i - target_tip_i||)
    )
```

混合后：

```text
L_i =
    alpha_i * L_tip_dir_vec_i
  + (1 - alpha_i) * L_full_hand_i
```

总目标：

```text
L(q) = sum_i L_i + norm_delta * ||q - q_prev||^2
```

这里使用 Huber loss，是为了在关键点噪声、遮挡、错误检测时比纯 L2 loss 更稳。

### 6.5 解析梯度

这个优化器的特点是手写了解析梯度，而不是用数值差分或 autograd。

对位置误差：

```text
e = robot_vec(q) - target_vec
d = ||e||

dL/dq = Huber'(d) * e / ||e|| * d(robot_vec)/dq
```

其中：

```text
d(robot_vec)/dq = J_task - J_origin
```

对单位方向误差，还需要归一化向量的 Jacobian：

```text
u = v / ||v||

du/dq = (I - u u^T) / ||v|| * dv/dq
```

这使得优化可以在实时控制中更快收敛。

## 7. 算法二：VectorOptimizer

`VectorOptimizer` 是更通用的关键向量优化器。

它把人手和机器人手都表示为一组向量，然后最小化对应向量之间的差。

默认 15 个关键向量为：

```text
每根手指:
palm_link -> finger{i}_link3
palm_link -> finger{i}_link4
palm_link -> finger{i}_tip_link
```

对应 MediaPipe：

```text
wrist -> PIP
wrist -> DIP
wrist -> TIP
```

YAML 配置示例：

```yaml
key_vectors:
  - {origin: palm_link, task: finger1_link3,    origin_kp: 0, task_kp: 2, scale: 1.0}
  - {origin: palm_link, task: finger1_link4,    origin_kp: 0, task_kp: 3, scale: 1.0}
  - {origin: palm_link, task: finger1_tip_link, origin_kp: 0, task_kp: 4, scale: 1.0}
```

目标向量：

```text
target_vec_i = scale_i * (keypoints[task_kp_i] - keypoints[origin_kp_i])
```

机器人向量：

```text
robot_vec_i(q) = FK(task_link_i, q) - FK(origin_link_i, q)
```

Loss：

```text
L(q) =
    mean_i Huber(||robot_vec_i(q) - target_vec_i||)
  + norm_delta * ||q - q_prev||^2
```

VectorOptimizer 的优势是配置灵活，换手型、换设备、换目标 link/keypoint 映射时比较方便。缺点是它没有 AdaptiveOptimizerAnalytical 中的 pinch-aware 模式切换，面对精细捏合动作时表达能力较弱。

## 8. 低通滤波

`Retargeter` 在优化输出后默认应用一阶低通滤波：

```text
y_t = y_{t-1} + alpha * (x_t - y_{t-1})
```

其中：

```yaml
retarget:
  lp_alpha: 0.2
```

`alpha` 越小，越平滑，但延迟越大。

## 9. 支持的硬件设备

### 9.1 输入设备

| 输入源 | 支持方式 | 依赖 |
| --- | --- | --- |
| Apple Vision Pro | `avp_stream` 获取手部 25 关节矩阵，再转 MediaPipe 21 点 | `avp_stream` |
| MP4 视频 | OpenCV 读取视频，MediaPipe Hands 检测 | `opencv-python`, `mediapipe` |
| Intel RealSense | RGB stream + MediaPipe Hands | `pyrealsense2`, `opencv-python`, `mediapipe` |
| STEREOLABS ZED | 左目 RGB + MediaPipe Hands | `pyzed`, `opencv-python`, `mediapipe` |
| 录制回放 | `.pkl` 帧序列 | Python pickle |
| Manus glove | 本仓库不直接采集；需外部 ROS2/Manus SDK 管线转成兼容关键点 | `wuji-hand-teleop` |

### 9.2 输出设备

| 输出目标 | 控制方式 |
| --- | --- |
| MuJoCo 仿真 | `data.ctrl[:] = qpos` |
| Wuji Hand 真机 | `wujihandpy.Hand().realtime_controller()` |

真实硬件控制会把 `(20,)` 输出 reshape 成 `(5, 4)`。

## 10. 当前算法特点

### 优点

1. 不需要训练数据，依赖 URDF 和几何约束即可运行。
2. 可解释性强，每个 loss 项都有明确物理意义。
3. 支持关节限位，天然避免超出机器人机械范围。
4. 使用上一帧 warm start，适合实时连续输入。
5. AdaptiveOptimizerAnalytical 对捏合动作做了专门处理。
6. VectorOptimizer 配置灵活，便于快速适配不同关键点映射。

### 局限

1. 依赖输入关键点质量；单目 MediaPipe 深度不准时需要较多经验调参。
2. 优化目标是逐帧的，虽然有 `norm_delta` 和低通滤波，但没有显式时间序列理解。
3. 不直接学习人手到机器人手的复杂非线性映射，某些姿态可能需要手工调 `segment_scaling`、`w_pos`、`w_dir`、`pinch_thresholds`。
4. 捏合检测依赖拇指和其他指尖距离，不能理解任务语义。
5. 每帧运行非线性优化，实时性受 CPU、URDF 复杂度和优化收敛影响。
6. 对接触、力控、物体交互没有建模。

## 11. 下一步：基于学习式模型的 retargeting

后续可以在当前几何优化系统上叠加学习式模型，而不是完全替换现有系统。更稳妥的方向是把当前优化器作为 teacher、约束器或安全后端，让学习模型负责更快、更鲁棒的初值预测和时序表达。

### 11.1 为什么引入学习式模型

学习式模型可以补足当前方法的几个短板：

1. 从噪声关键点中恢复更稳定的手势意图。
2. 学习不同输入设备之间的系统性偏差，例如 Vision Pro、单目视频、RealSense 的尺度和深度差异。
3. 学习人手到机器人手之间非线性的形态映射，减少手工调参。
4. 用时间序列模型减少抖动和延迟之间的矛盾。
5. 以更低推理成本替代或加速每帧非线性优化。
6. 为接触、抓取稳定性、任务成功率引入数据驱动目标。

### 11.2 推荐的模型输入

基础输入仍然建议保持 MediaPipe 风格，因为仓库已经把所有设备统一到了 `(21, 3)`：

```python
X_t = {
    "keypoints": np.ndarray,      # (21, 3)
    "hand_side": int,             # left/right
    "prev_qpos": np.ndarray,      # (20,)
    "confidence": np.ndarray,     # optional, (21,) or scalar
}
```

为了让模型对坐标系和尺度更鲁棒，推荐加入派生特征：

```text
normalized keypoints in wrist frame
wrist -> joint vectors
finger segment directions
finger segment lengths
tip-to-tip distances
pinch distance features
previous qpos
previous velocity
```

对于时间序列模型，可输入最近 `T` 帧：

```python
X = np.ndarray  # shape = (T, 21, 3)
prev_qpos = np.ndarray  # shape = (T, 20)
```

### 11.3 推荐的模型输出

最直接输出：

```python
y = qpos  # shape = (20,)
```

更好的输出可以包含不确定性或残差：

```python
{
    "qpos_pred": np.ndarray,       # (20,)
    "confidence": float,
    "residual": np.ndarray,        # optional, (20,)
}
```

还有一种稳健做法是让模型输出优化器初值：

```text
model(keypoints) -> init_qpos
NLopt(init_qpos) -> final_qpos
```

这样模型不必完全承担安全约束，最终仍由 URDF joint limits 和几何 loss 修正。

### 11.4 数据来源

可以逐步构建三类数据集。

#### 11.4.1 Teacher 数据

用当前优化器生成监督标签：

```text
输入关键点 -> 当前优化器 qpos
```

优点是成本低，可以快速生成大量数据。缺点是模型上限受 teacher 限制，学习到的是当前优化器的行为。

#### 11.4.2 人工标定/遥操作数据

采集操作者输入和真实 Wuji Hand 控制输出：

```text
input keypoints
retargeted qpos
manual correction qpos
task outcome
```

如果能记录人工修正或成功抓取动作，模型可以学到比几何优化更贴近真实任务的映射。

#### 11.4.3 仿真增强数据

在 MuJoCo 中生成机器人手动作，再反向构造或扰动人手关键点：

```text
robot qpos -> FK skeleton -> synthetic keypoints -> qpos
```

可以加入噪声、遮挡、尺度扰动、左右手镜像、深度压缩等增强。

### 11.5 模型路线

#### 阶段一：MLP 或小型 Transformer 回归器

输入单帧标准化关键点和上一帧 qpos，输出当前 qpos。

```text
features: (21, 3) + prev_qpos + pinch features
model: MLP / small Transformer encoder
output: (20,)
```

优点：

- 实现快。
- 推理成本低。
- 可以作为现有优化器的 warm start。

训练 loss：

```text
L = ||q_pred - q_teacher||^2
  + lambda_smooth * ||q_pred - q_prev||^2
  + lambda_limit * joint_limit_penalty
```

#### 阶段二：时序模型

使用最近多帧关键点：

```text
input: T frames of keypoints and previous qpos
model: TCN / GRU / lightweight Transformer
output: qpos_t
```

目标是降低噪声和抖动，同时减少低通滤波带来的延迟。

可以加入速度和加速度正则：

```text
L_vel = ||q_t - q_{t-1}||
L_acc = ||q_t - 2q_{t-1} + q_{t-2}||
```

#### 阶段三：Residual Retargeting

让模型学习当前几何优化器的残差：

```text
q_base = optimizer(keypoints)
delta_q = model(keypoints, q_base, history)
q_final = q_base + delta_q
```

这个路线很实用，因为保留了几何优化器的稳定性，同时让模型修正常见误差，例如特定手指过伸、单目深度偏差、捏合姿态不准。

#### 阶段四：任务感知模型

如果后续目标不仅是姿态模仿，而是抓取/操作任务成功率，可以加入任务信号：

```text
object pose
contact state
force/torque
tactile readings
success/failure labels
```

此时输出可以不再只是模仿人手，而是对机器人手更合适的抓取姿态。

## 12. 与现有优化器的集成方式

### 12.1 模型作为 warm start

推荐优先做这个方案：

```text
keypoints -> model -> init_qpos -> NLopt -> qpos
```

优点：

- 对现有代码侵入小。
- 保留 joint limits 和几何 loss。
- 如果模型失败，优化器仍可兜底。
- 可以减少 SLSQP 迭代次数，提高实时性。

需要修改的位置主要在 `BaseOptimizer.solve()` 的初始化逻辑附近，把 `init_qpos` 从模型预测得到。

### 12.2 模型作为直接控制器

```text
keypoints -> model -> qpos
```

优点是速度最快，缺点是安全性和泛化风险更高。必须加入：

```text
joint clipping
velocity limit
low-pass filter
confidence gate
fallback optimizer
```

建议只在充分验证后用于真机。

### 12.3 模型作为残差修正器

```text
keypoints -> optimizer -> q_base
(keypoints, q_base) -> model -> delta_q
q_final = q_base + delta_q
```

这适合在已有优化结果基本正确，但某些姿态存在系统误差时使用。

## 13. 评估指标

学习式模型不能只看 `qpos` MSE，还应保留几何和任务指标。

推荐指标：

```text
joint angle MSE
tip position error
tip direction error
full hand vector error
pinch distance error
temporal jitter
latency
joint limit violation rate
optimization fallback rate
real robot task success rate
```

对于真机，还应重点看：

```text
最大关节速度
指尖抖动
异常姿态次数
急停/限位触发次数
抓取成功率
```

## 14. 推荐实施计划

### Step 1：建立数据记录格式

在现有 teleop loop 中记录：

```python
{
    "t": float,
    "raw_keypoints": np.ndarray,       # (21, 3)
    "transformed_keypoints": np.ndarray, # (21, 3)
    "qpos_teacher": np.ndarray,        # (20,)
    "qpos_filtered": np.ndarray,       # (20,)
    "cost": float,
    "pinch_alphas": np.ndarray,        # (5,)
    "hand_side": str,
    "input_device": str,
}
```

### Step 2：训练 teacher imitation 模型

先用当前优化器生成标签，训练单帧 MLP。

目标不是马上替换优化器，而是验证：

```text
模型能否以低误差复现 optimizer 输出
模型作为 init_qpos 能否减少优化迭代
模型输出是否比当前 warm start 更稳定
```

### Step 3：接入 warm start

新增可选配置：

```yaml
learning:
  enabled: true
  mode: warm_start
  model_path: path/to/model.onnx
  fallback_to_midpoint: true
```

推理建议优先用 ONNX Runtime，便于部署和控制依赖。

### Step 4：加入时序模型

当单帧模型稳定后，引入 TCN/GRU/Transformer，使用最近若干帧降低抖动。

### Step 5：真机安全验证

真机验证前必须保留：

```text
joint limit clipping
velocity limiting
low-pass filter
confidence threshold
zero/invalid input handling
optimizer fallback
emergency stop
```

## 15. 总结

当前 `wuji-retargeting` 是一个清晰的几何优化式 retargeting 系统：

```text
MediaPipe 21 点
  -> wrist frame 标准化
  -> TipDirVec / FullHandVec / key vectors
  -> Pinocchio FK/Jacobian
  -> NLopt SLSQP
  -> 20 维 Wuji Hand qpos
```

`AdaptiveOptimizerAnalytical` 更适合实时遥操作和捏合动作，`VectorOptimizer` 更适合可配置映射和快速实验。

下一步如果引入学习式模型，建议不要一开始直接端到端替代现有优化器。更稳健的路线是：

```text
阶段 1：学习模型作为 optimizer warm start
阶段 2：学习模型作为 residual correction
阶段 3：时序模型降低抖动和延迟
阶段 4：结合接触/任务数据做 task-aware retargeting
```

这样可以保留当前系统的可解释性和机械安全约束，同时逐步获得学习模型的速度、鲁棒性和任务适应能力。
