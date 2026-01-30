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
