# 📘 README (English Version)
  
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
# What the CSV logs contain (field meanings)

Each run writes results into `results/<timestamp>/` and produces **two CSV files per scene**:

1.  **Candidate-level CSV**: `<scene_name>.csv`
    *   **One row per grasp candidate** (one left/right grasp pair).
    *   Contains grasp-point data + final metrics.

2.  **Stage-level CSV**: `<scene_name>_stage_metrics.csv`
    *   **Exactly 5 rows per candidate** (Stage1\~Stage5).
    *   If a stage fails, the remaining stages are still written and marked as `skipped_due_to_failure(...)`.

***

## 1) `<scene_name>.csv` (candidate-level) field meanings

### A. Identifiers

*   **scene**: scene name (typically the `scene_*.xml` filename without extension)
*   **object\_body**: the object body name (the freejoint object body)
*   **candidate\_id**: candidate index (starts at 0)
*   **pair\_dist**: distance between left/right grasp points in **world frame** (meters)

***

### B. Grasp point positions (world frame)

Positions of the grasp points in world coordinates (meters):

*   **left\_x\_w, left\_y\_w, left\_z\_w**
*   **right\_x\_w, right\_y\_w, right\_z\_w**

***

### C. Grasp point offsets relative to the AABB center

Offsets from the object AABB geometric center (meters):

*   **left\_x\_center, left\_y\_center, left\_z\_center**
*   **right\_x\_center, right\_y\_center, right\_z\_center**

> Computed as `*_center = *_w - center_w`, where `center_w` is the AABB center in world frame.

***

### D. Grasp point positions in the object/body frame

World-to-object transformed grasp points (meters):

*   **left\_x\_obj, left\_y\_obj, left\_z\_obj**
*   **right\_x\_obj, right\_y\_obj, right\_z\_obj**

> These coordinates are more stable across object motion (useful for reproducibility).

***

### E. Approach direction (world frame)

The code uses the grasp frame rotation matrix **Z-axis** as the approach direction (unitless direction vector):

*   **left\_ax\_w, left\_ay\_w, left\_az\_w**
*   **right\_ax\_w, right\_ay\_w, right\_az\_w**

***

### F. Final evaluation metrics

Returned by `evaluate_candidate()` as `metrics`:

*   **success**: success flag (1=success, 0=failure)

*   **has\_contact**: whether fingertip–object contact exists after closing (1/0)

*   **lift\_height**: lifted height (m), typically Δz from the initial object height

*   **hold\_duration**: hold duration (s)

*   **max\_slip**: maximum slip distance (m)
    > Computed from the change of object-to-gripper relative transforms over time (max of left/right).

*   **obj\_v\_rms**: RMS linear velocity of the object (m/s)

*   **obj\_w\_rms**: RMS angular velocity of the object (rad/s)
    > From MuJoCo `data.cvel`, aggregated over the hold phase.

*   **contact\_ratio**: fraction of hold frames with fingertip–object contact (0\~1)

*   **min\_contact\_dist**: minimum contact distance (m)
    > From MuJoCo contact `c.dist` (near 0 or negative when in contact/penetration).

*   **rel\_pos\_err**: relative position error between the two end-effectors (m)

*   **rel\_rot\_err\_deg**: relative rotation error between the two end-effectors (degrees)
    > The target relative transform is captured at the “grasped” moment (closed-chain reference).

*   **contact\_force\_n\_mean**: mean norm of contact force (Newtons, N)

*   **contact\_force\_n\_count**: number of valid samples used for the force average
    > Force extracted via `mj_contactForce` and averaged per stage.

*   **fail\_stage**: stage name where the failure occurred (e.g., `Stage3_close`, `Stage4_lift`, `exception`)

*   **fail\_reason**: failure reason string (e.g., `no_contact`, `dropped`, `ik_fail:...`, `viewer_closed`)

***

## 2) `<scene_name>_stage_metrics.csv` (stage-level) field meanings

This CSV logs **5 rows per candidate** for:

1.  `Stage1_pregrasp`
2.  `Stage2_grasp`
3.  `Stage3_close`
4.  `Stage4_lift`
5.  `Stage5_hold`

### A. Identifiers & stage info

*   **scene**: scene name
*   **object\_body**: object body name
*   **candidate\_id**: candidate index
*   **stage\_idx**: stage index (1\~5)
*   **stage**: stage name string
*   **stage\_ok**: stage success flag (1=ok, 0=failure/skipped)
*   **fail\_reason**: reason string
    *   For auto-filled remaining stages after failure: `skipped_due_to_failure(StageX:reason)`

***

### B. Time and object pose snapshot

*   **sim\_time**: simulation time at the moment the row is written (s)
*   **obj\_x, obj\_y, obj\_z**: object position in world frame (m)

***

### C. Stage metrics (may be NaN depending on stage)

These match the candidate-level metrics, but are **stage-specific snapshots/statistics**. If a metric is not applicable for a stage, it is written as `NaN` (per your defaults):

*   **lift\_height** (m)
*   **hold\_duration** (s)
*   **max\_slip** (m)
*   **obj\_v\_rms** (m/s)
*   **obj\_w\_rms** (rad/s)
*   **contact\_ratio** (0\~1)
*   **min\_contact\_dist** (m)
*   **rel\_pos\_err** (m)
*   **rel\_rot\_err\_deg** (deg)
*   **contact\_force\_n\_mean** (N)
*   **contact\_force\_n\_count** (count)

***

