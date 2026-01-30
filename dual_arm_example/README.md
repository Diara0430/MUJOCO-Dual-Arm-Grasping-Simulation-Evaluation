# 📘 README（中文）

# 双臂抓取评估系统（Scheme B）

本项目提供了使用 **MuJoCo + Mink** 进行双 UR5 机械臂对物体抓取的完整评估流程（Grasp Pair Evaluation Scheme B）。系统通过对候选双臂抓取点进行逐阶段仿真，输出抓取成功率、滑移、接触、姿态误差等多项指标。

***

## 📂 项目结构

下图展示了项目的整体目录结构（部分省略）：

    ├── assets/dual_arm_and_single_arm      # 机械臂、底座、夹爪、场景与物体XML文件
    ├── data                                 # 抓取轨迹数据（程序自动生成）
    ├── EnvironmentAndObjects                # 机械臂与物体的 Python 类
    ├── utils                                # 工具代码（可视化、插值、转换等）
    ├── tools                                # 生成与修补 MuJoCo 场景的工具脚本
    ├── results                              # 每次运行自动生成结果目录
    ├── batch_grasp_pair_eval_schemeB.py     # ★ 主程序：双臂抓取评估 Scheme B
    ├── grasp_pair_evaluation_ur5.py         # 针对 UR5 的抓取评估逻辑（旧版本）
    ├── load_ur5.py                          # UR5 加载与预处理
    ├── trans_single_arm_2_dual.py           # 单→双臂协调示例
    └── README.md

### assets/

用于构建场景的所有资源，包括：

*   **UR5 双臂模型**（`dual_ur5.xml`, `dual_ur5_scene.xml`）
*   **机械臂、夹爪、底座的 mesh 文件**
*   **场景文件（scene\_\*.xml）**
*   **物体 XML （obj\_big\_box.xml）**

### data/

程序运行过程中自动生成的轨迹数据 `.npz`。

### EnvironmentAndObjects/

包含核心类：

*   `robot_arm.py`：封装单个机械臂在 Mink/MuJoCo 下的控制接口
*   `scene_object.py`：场景物体定义

### utils/

可复用工具，如：

*   `trajectory_visualization.py`
*   `common_utils.py`
*   `transfer_from_small_2_big.py`

### tools/

辅助脚本：

*   `generate_scenes.py`：自动生成 scene xml
*   `patch_match_paths.py`：修复路径

***

## 🚀 主程序功能：batch\_grasp\_pair\_eval\_schemeB.py

该脚本实现 **双臂从地面抓取（Scheme B）** 的完整评估流程，包括：

### 1. 自动加载场景

从 `assets/dual_arm_and_single_arm/scenes/scene_*.xml` 加载所有场景。

### 2. 自动发现物体

自动检测场景中 **freejoint 物体**（需有一个自由关节物体）。

### 3. AABB 自适应采样抓取对

基于物体几何包围盒（AABB）：

*   采样候选抓取点对
*   自动生成左右手朝向（approach vectors）
*   转换到
    *   世界坐标
    *   几何中心坐标
    *   物体坐标系

### 4. 五阶段评估流程

每个抓取对执行五个阶段（固定写入 5 行 CSV）：

| Stage | 含义                 |
| ----- | ------------------ |
| 1     | Pregrasp：移动至预抓取位姿  |
| 2     | Grasp：手爪对准抓取点      |
| 3     | Close：闭合手爪建立接触     |
| 4     | Lift：双臂协同抬起物体      |
| 5     | Hold：保持姿态，检测滑移与稳定性 |

### 5. 输出指标包括：

*   成功与否
*   抬升高度
*   最大滑移
*   物体 RMS 速度 / 角速度
*   相对姿态误差（闭链）
*   接触率
*   指尖 - 物体最小距离
*   接触力统计

所有候选抓取都会写入：

    results/20260126_xxxxxx/
        ├── scene_big_box.csv           # 每个 candidate 一行
        ├── scene_big_box_stage_metrics.csv   # 每个 candidate 5 行
        ├── top5_scene_big_box.json     # 按成功+稳定性排序
        └── summary_all.json

***

## ▶️ 如何运行 Scheme B 双臂抓取评估

确保安装依赖（MuJoCo、mink、numpy、scipy 等）。

然后直接运行：

```bash
python batch_grasp_pair_eval_schemeB.py
```

运行后将自动打开 MuJoCo Viewer。

### ⌨️ Viewer 操控键位

| 按键    | 功能                |
| ----- | ----------------- |
| q     | 退出                |
| space | 暂停/继续             |
| .     | 单步执行              |
| n     | 跳过当前 candidate    |
| v     | 显示/隐藏抓取可视化 marker |

***

## ▶️ 单臂 / 双臂执行示例（另一个 demo）

如果你想看 trajectory 的执行（非抓取评估）：

```bash
python trans_single_arm_2_dual.py
```

其中两个函数对应：

*   `execute_trajectory_general` → 单臂执行 trajectory
*   `execute_object_centric_trajectory` → 双臂物体中心轨迹执行

***

## 🔨 自动生成场景

将一个 obj xml 自动生成对应的 scene：

```bash
python tools/generate_scenes.py
```

***

# 📘 README (English)

# Dual-Arm Grasp Pair Evaluation System (Scheme B)

This project provides a complete evaluation pipeline for **dual UR5 robot arms** performing **ground grasping (Scheme B)** using **MuJoCo + Mink**.  
It samples grasp-pair candidates, runs multi-stage simulations, and outputs success rate, slip metrics, contact statistics, and relative pose errors.

***

## 📂 Project Structure

The project is organized as follows:

    ├── assets/dual_arm_and_single_arm      # Robot models, meshes, scenes
    ├── data                                 # Auto-generated trajectory data
    ├── EnvironmentAndObjects                # Robot and object classes
    ├── utils                                # Utilities (visualization, conversions)
    ├── tools                                # Scene generation scripts
    ├── results                              # Evaluation outputs
    ├── batch_grasp_pair_eval_schemeB.py     # ★ Main program
    └── trans_single_arm_2_dual.py           # Single → dual arm demo

### assets/

Contains:

*   Dual UR5 model XML files
*   All meshes (UR5, Robotiq gripper, base stand, Panda model, etc.)
*   Scene XMLs
*   Object XML files

### EnvironmentAndObjects/

Core components:

*   `robot_arm.py` — UR5 control helper for Mink & MuJoCo
*   `scene_object.py` — Object class

### utils/

Reusable tools:

*   trajectory visualization
*   trajectory scaling
*   common utilities

### tools/

Scene generation & path patching tools.

***

## 🚀 Main Program: batch\_grasp\_pair\_eval\_schemeB.py

This script implements full **dual-arm ground grasp evaluation (Scheme B)**.

### Main features:

### 1. Automatic scene loading

From `assets/.../scenes/scene_*.xml`.

### 2. Automatic object discovery

Finds the **single freejoint object** in the scene.

### 3. Adaptive grasp-pair sampling

Based on the object AABB:

*   sample two grasp points
*   compute approach direction
*   convert to world / center / object coordinates

### 4. Five-stage evaluation

| Stage | Meaning                             |
| ----- | ----------------------------------- |
| 1     | Pregrasp pose                       |
| 2     | Grasp pose alignment                |
| 3     | Close gripper and establish contact |
| 4     | Lift the object                     |
| 5     | Hold and measure slip/stability     |

### 5. Per-candidate metrics:

*   success / failure reasons
*   lift height
*   slip amount
*   RMS linear & angular velocity
*   closed-chain relative pose error
*   contact ratio
*   minimum finger–object distance
*   contact force stats

Results are written into:

    results/20260126_xxxxxx/
        ├── scene_xxx.csv
        ├── scene_xxx_stage_metrics.csv
        ├── top5_scene_xxx.json
        └── summary_all.json

***

## ▶️ Run the Evaluation

```bash
python batch_grasp_pair_eval_schemeB.py
```

### Keyboard Controls (MuJoCo Viewer)

| Key   | Description          |
| ----- | -------------------- |
| q     | Quit                 |
| space | Pause / resume       |
| .     | Step forward         |
| n     | Skip candidate       |
| v     | Toggle grasp markers |

***

## ▶️ Single-arm / Dual-arm Execution Demo

To visualize single-arm or dual-arm trajectory playback:

```bash
python trans_single_arm_2_dual.py
```

Functions:

*   `execute_trajectory_general` → single-arm
*   `execute_object_centric_trajectory` → dual-arm coordinated motion

***

## 🔨 Generate Scene Files Automatically

```bash
python tools/generate_scenes.py
```
