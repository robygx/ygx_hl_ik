# PiM-IK: Physics-informed Mamba Inverse Kinematics

> 基于物理内化约束和 Mamba 时序建模的 7-DOF 机械臂逆运动学求解器

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)]()

---

## 📚 目录

- [项目概述](#项目概述)
- [核心功能](#核心功能)
- [网络架构](#网络架构)
- [数据集](#数据集)
- [训练与评估](#训练与评估)
- [消融实验](#消融实验)
- [VR 遥操作集成](#vr-遥操作集成)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [实验结果](#实验结果)

---

## 项目概述

**PiM-IK** 是一个用于求解 7-DOF 机械臂逆运动学的深度学习方法。该方法结合了：

1. **Mamba 时序建模** - 捕捉运动的时间相关性，提升轨迹平滑度
2. **物理内化约束** - 融入机器人运动学约束（Swivel 角、肘部位置、平滑性）
3. **6D 连续位姿表征** - 避免四元数/欧拉角的奇异性问题

### 解决的问题

- ✅ **传统 IK 求解慢** - 神经网络前向推理 < 1ms
- ✅ **轨迹不平滑** - Mamba 时序建模显著降低 Jerk
- ✅ **关节限位约束** - 物理内化损失确保解在可行域内
- ✅ **VR 遥操作 OOD** - 工作空间映射防止分布外输入

---

## 核心功能

### 1. 神经网络 IK 求解

```python
from pim_ik_net import PiM_IK_Net

model = PiM_IK_Net(d_model=256, num_layers=4)
T_ee = ...  # (B, W, 4, 4) 末端位姿序列
pred_swivel = model(T_ee)  # (B, W, 2) 预测臂角 [cos(φ), sin(φ)]
```

### 2. 分层 IK 求解器

结合神经网络和几何约束的分层优化：

```python
from inference import HierarchicalIKSolver, TargetGenerator

# Task 1: 末端位姿跟踪 (Primary)
# Task 2: 肘部位置跟踪 (Secondary, 投影到零空间)
```

### 3. VR 工作空间映射

```python
from workspace_retargeter import WorkspaceRetargeter

retargeter = WorkspaceRetargeter(
    vr_json_path='vr_workspace_limits.json',
    dataset_json_path='dataset_workspace_limits.json',
    uniform_scale=True
)
T_mapped = retargeter.map_pose(T_vr)
```

---

## 网络架构

### PiM-IK_Net 结构

```
输入: T_ee (B, W, 4, 4) 齐次变换矩阵序列
         ↓
    [9D 转换] 平移(3) + 6D旋转(6) = 9D连续表征
         ↓
    [Stem] Linear + ReLU
         ↓
    [Conv1d] d_conv=4, 扩张时序感受野
         ↓
  [Mamba×4] 状态空间模型, d_state=16
         ↓
    [Head] Linear → L2归一化
         ↓
输出: swivel (B, W, 2) [cos(φ), sin(φ)]
```

### 物理内化损失

$$\mathcal{L} = w_{swivel} \mathcal{L}_{swivel} + w_{elbow} \mathcal{L}_{elbow} + w_{smooth} \mathcal{L}_{smooth}$$

| 损失项 | 公式 | 作用 |
|--------|------|------|
| $\mathcal{L}_{swivel}$ | $\|pred - gt\|_1$ | 拟人先验约束 |
| $\mathcal{L}_{elbow}$ | $\|p_e(pred) - p_e(gt)\|_2^2$ | 三维空间约束 |
| $\mathcal{L}_{smooth}$ | $\|\phi_t - 2\phi_{t-1} + \phi_{t-2}\|_2^2$ | 时序平滑约束 |

---

## 数据集

### 训练数据

| 数据集 | 样本量 | 来源 |
|--------|--------|------|
| ACCAD_CMU | 3,614,699 帧 | 运动捕捉数据 |
| GRAB | 1,618,048 帧 | 手部操作数据 |

### 数据格式

```python
data = np.load('ACCAD_CMU_merged_training_data_with_swivel.npz')
# {
#   'X': (N, 10, 14),           # 时序窗口输入
#   'T_ee': (N, 4, 4),           # 末端执行器齐次变换矩阵
#   'swivel_angle': (N, 2),      # 臂角 [cos(φ), sin(φ)]
#   'joint_positions': (N, 28),  # 关节位置
#   'L_upper': (N,),             # 上臂长度
#   'L_lower': (N,),             # 前臂长度
#   'is_valid': (N,),            # 有效性掩码
# }
```

---

## 训练与评估

### 训练命令

```bash
# 标准训练 (W=30, 完整损失)
torchrun --nproc_per_node=2 trainer.py --epochs 50

# 自定义参数
torchrun --nproc_per_node=2 trainer.py \
    --window_size 30 \
    --w_swivel 1.0 \
    --w_elbow 1.0 \
    --w_smooth 0.1 \
    --epochs 50
```

### 评估命令

```bash
# 单模型评估
python evaluate.py \
    --checkpoint checkpoints/20260227_111614/best_model.pth \
    --data_path /data0/wwb_data/ygx_data/data_ygx_pose+dof/GRAB_training_data_with_swivel.npz
```

---

## 消融实验

### 1. 时序窗口大小消融

验证 Mamba 时序建模对平滑度的贡献：

```bash
python ablation_window_size.py \
    --data_path /data0/wwb_data/ygx_data/data_ygx_pose+dof/ACCAD_CMU_merged_training_data_with_swivel.npz \
    --num_frames 1000
```

| 模型 | MAE (°) | Jerk | 说明 |
|------|---------|------|------|
| W=30 | ~6.5 | 最低 | 完整时序建模 |
| W=15 | ~6.0 | 中等 | 减小时序窗口 |
| W=1 | ~6.3 | 最高 | 无记忆基线 (MLP) |

### 2. 物理内化损失消融

验证各损失组件的有效性：

```bash
python ablation_loss_eval.py
```

| 配置 | $\mathcal{L}_{swivel}$ | $\mathcal{L}_{elbow}$ | $\mathcal{L}_{smooth}$ | 说明 |
|------|:----------------------:|:--------------------:|:--------------------:|------|
| Baseline | 1.0 | 0.0 | 0.0 | 纯数据驱动 |
| Variant A | 1.0 | 1.0 | 0.0 | +空间约束 |
| Ours | 1.0 | 1.0 | 0.1 | 完整损失 |

### 3. 网络层数消融

```bash
python ablation_layers_eval.py
```

---

## VR 遥操作集成

### 工作空间分析

```bash
# 1. 分析训练数据集工作空间
python analyze_workspace.py

# 2. 在 VR 端采集控制器工作空间
cd VR_data_any
python record_vr_workspace.py

# 3. 对比两个工作空间
python compare_workspaces.py
```

### 集成到遥操作代码

```python
from workspace_retargeter import WorkspaceRetargeter

# 初始化
retargeter = WorkspaceRetargeter(
    vr_json_path='vr_workspace_limits.json',
    dataset_json_path='dataset_workspace_limits.json'
)

# 在控制循环中
T_mapped = retargeter.map_pose(teleData.left_wrist_pose)
```

详见：[workspace_retargeting_README.md](workspace_retargeting_README.md)

---

## 项目结构

```
ygx_hl_ik/
├── pim_ik_net.py              # PiM-IK 网络定义
├── pim_ik_kinematics.py       # 运动学层和损失函数
├── trainer.py                 # 分布式训练脚本
├── inference.py               # 推理和 IK 求解
├── evaluate.py                # 评估脚本
│
├── ablation_window_size.py    # 窗口大小消融
├── ablation_loss_eval.py      # 损失消融
├── ablation_layers_eval.py    # 层数消融
│
├── analyze_workspace.py       # 工作空间分析
├── workspace_retargeter.py    # VR 映射工具
├── compare_workspaces.py      # 工作空间对比
│
├── dataset_workspace_limits.json  # 数据集边界统计
└── checkpoints/               # 模型权重
```

---

## 快速开始

### 环境安装

```bash
# 创建 conda 环境
conda create -n pim_ik python=3.10
conda activate pim_ik

# 安装依赖
pip install torch numpy scipy pinocchio mamba-ssm causal-conv1d
pip install matplotlib wandb
```

### 下载数据

```bash
python download_hf_dataset.py
```

### 训练模型

```bash
torchrun --nproc_per_node=2 trainer.py --epochs 50
```

### VR 遥操作

```bash
cd hl_ik_xr_tele/teleop/robot_control
python3 robot_arm_ik_nn_ygx.py --input-mode vr
```

---

## 实验结果

### 工作空间统计

| 轴 | 1% 分位 | 99% 分位 | 活动范围 |
|----|---------|----------|----------|
| X | -0.146 m | 0.263 m | 0.409 m |
| Y | 0.012 m | 0.488 m | 0.476 m |
| Z | -0.069 m | 0.328 m | 0.397 m |

### VR 映射参数

| 参数 | 值 |
|------|-----|
| 统一缩放系数 | 0.454x |
| 中心偏移 | 0.267 m |

### 模型性能

| 指标 | 值 |
|------|-----|
| 推理延迟 | < 1ms (GPU) |
| Swivel MAE | ~6° |
| 肘部位置误差 | ~10mm |

---

## 引用

如果本项目对你有帮助，请考虑引用：

```bibtex
@software{pim_ik_2025,
  title={PiM-IK: Physics-informed Mamba Inverse Kinematics for 7-DOF Robotic Arms},
  author={Your Name},
  year={2025},
  url={https://github.com/robygx/ygx_hl_ik}
}
```

---

## 许可证

MIT License

---

## 更新日志

- **2025-02-28**: 添加工作空间映射和 VR 遥操作集成
- **2025-02-27**: 完成时序窗口消融实验
- **2025-02-26**: 实现物理内化损失函数
- **2025-02-25**: 初始版本，基础 PiM-IK 网络
