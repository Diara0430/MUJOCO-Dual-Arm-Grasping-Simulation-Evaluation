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

# CSV 记录字段说明

该程序每次运行会在 `results/<时间戳>/` 下输出 **两类 CSV**：

1.  **候选抓取汇总 CSV**：`<scene_name>.csv`
    *   **每一行**对应一个 grasp candidate（一个候选双手抓取点对）的评估结果（包含抓取点信息 + 最终指标）。

2.  **阶段指标 CSV**：`<scene_name>_stage_metrics.csv`
    *   **每个 candidate 固定写 5 行**（Stage1\~Stage5），记录每个阶段开始/结束后的状态与指标。
    *   若某阶段失败，会把后续阶段补齐并标记为 `skipped_due_to_failure(...)`。

***

## 1) `<scene_name>.csv`（candidate-level）字段含义

### A. 基本标识

*   **scene**：场景名称（通常来自 `scene_*.xml` 的文件名去掉后缀）
*   **object\_body**：场景里被抓取的物体 body 名称（freejoint 对应的 body）
*   **candidate\_id**：候选抓取对的编号（从 0 开始）
*   **pair\_dist**：左右抓取点在**世界坐标系**下的距离（米，m）

***

### B. 抓取点位置（世界坐标系 world frame）

以下字段描述 candidate 的左右抓取点在世界系中的坐标（单位：m）：

*   **left\_x\_w, left\_y\_w, left\_z\_w**：左抓取点世界坐标
*   **right\_x\_w, right\_y\_w, right\_z\_w**：右抓取点世界坐标

***

### C. 抓取点位置（相对 AABB 几何中心）

这是“抓取点相对物体 AABB 几何中心”的偏移量（世界系下计算，单位：m）：

*   **left\_x\_center, left\_y\_center, left\_z\_center**
*   **right\_x\_center, right\_y\_center, right\_z\_center**

> 含义：`*_center = *_w - center_w`，其中 `center_w` 是物体 AABB 的中心点（世界坐标）。

***

### D. 抓取点位置（物体坐标系 object/body frame）

将抓取点从世界系变换到**物体局部坐标系**（单位：m）：

*   **left\_x\_obj, left\_y\_obj, left\_z\_obj**
*   **right\_x\_obj, right\_y\_obj, right\_z\_obj**

> 含义：如果物体发生移动/旋转，物体系下的抓取点更“稳定”，便于复现与分析。

***

### E. 抓取“接近方向”（approach direction，世界系）

代码用抓取姿态的旋转矩阵 `rot_w` 的 **Z 轴**作为接近方向（单位无，方向向量）：

*   **left\_ax\_w, left\_ay\_w, left\_az\_w**：左抓取姿态 Z 轴在世界系的分量
*   **right\_ax\_w, right\_ay\_w, right\_az\_w**：右抓取姿态 Z 轴在世界系的分量

> 你可以理解为：夹爪“朝向/插入方向”的单位向量。

***

### F. 最终评估结果与指标（metrics）

这些字段来自 `evaluate_candidate()` 的最终返回 `metrics`：

*   **success**：是否成功（1=成功，0=失败）
    > 当前实现中 success 近似由“抬升高度达到阈值且保持阶段有持续时间”等条件综合决定。

*   **has\_contact**：是否在关爪后建立了指尖-物体接触（1/0）

*   **lift\_height**：抬升高度（m），通常是物体 z 方向相对初始高度的增量

*   **hold\_duration**：保持阶段持续时间（s）

*   **max\_slip**：保持阶段内最大滑移量（m）
    > 计算方式：物体坐标系到夹爪的相对位姿在时间上的平移变化（左右手取最大值）。

*   **obj\_v\_rms**：物体线速度 RMS（m/s）

*   **obj\_w\_rms**：物体角速度 RMS（rad/s）
    > 来自 MuJoCo 的 `data.cvel`，在保持阶段统计均方根。

*   **contact\_ratio**：保持阶段中“有指尖-物体接触”的帧比例（0\~1）

*   **min\_contact\_dist**：接触距离最小值（m）
    > 来自 MuJoCo contact 的 `c.dist`，数值越小表示越接触/嵌入越多（接触时可能接近 0 或为负）。

*   **rel\_pos\_err**：左右末端执行器之间的**相对位置误差**（m）

*   **rel\_rot\_err\_deg**：左右末端执行器之间的**相对旋转误差**（deg）
    > 用闭链约束：以“夹住物体瞬间”的左右手相对位姿作为目标，后续 lift/hold 计算偏差。

*   **contact\_force\_n\_mean**：接触力范数的平均值（N）

*   **contact\_force\_n\_count**：统计接触力时的样本计数（用于平均的有效次数/帧数）
    > 代码中用 `mj_contactForce` 得到接触力（取前三维力的范数），然后在阶段内做平均。

*   **fail\_stage**：失败发生在哪个阶段（如 `"Stage3_close"` / `"Stage4_lift"` / `"exception"` 等）

*   **fail\_reason**：失败原因字符串（如 `no_contact`, `dropped`, `ik_fail:...`, `viewer_closed` 等）

***

## 2) `<scene_name>_stage_metrics.csv`（stage-level）字段含义

该 CSV **每个 candidate 固定 5 行**，对应：

1.  `Stage1_pregrasp`：到预抓取位姿
2.  `Stage2_grasp`：到抓取位姿
3.  `Stage3_close`：关爪并检测接触
4.  `Stage4_lift`：抬升
5.  `Stage5_hold`：保持并统计稳定性

### A. 基本标识与阶段信息

*   **scene**：场景名称
*   **object\_body**：物体 body 名
*   **candidate\_id**：candidate 编号
*   **stage\_idx**：阶段序号（1\~5）
*   **stage**：阶段名称（如 `Stage3_close`）
*   **stage\_ok**：该阶段是否成功（1=成功，0=失败/跳过）
*   **fail\_reason**：失败/跳过原因
    *   若是被补齐的后续阶段，会形如：`skipped_due_to_failure(StageX:reason)`

***

### B. 时刻与物体位置（阶段记录时刻的快照）

*   **sim\_time**：写入该行时的仿真时间（s）
*   **obj\_x, obj\_y, obj\_z**：写入该行时物体位置（世界坐标，m）

***

### C. 阶段内/阶段后统计指标（字段和 candidate-level 基本一致）

以下字段在不同阶段可能是 NaN 或仅部分填写（你代码里通过默认值 `np.nan` 填充未适用项）：

*   **lift\_height**（m）
*   **hold\_duration**（s）
*   **max\_slip**（m）
*   **obj\_v\_rms**（m/s）
*   **obj\_w\_rms**（rad/s）
*   **contact\_ratio**（0\~1）
*   **min\_contact\_dist**（m）
*   **rel\_pos\_err**（m）
*   **rel\_rot\_err\_deg**（deg）
*   **contact\_force\_n\_mean**（N）
*   **contact\_force\_n\_count**（count）

> 例：Stage3 会记录 `min_contact_dist` 与接触力统计；Stage4 会记录 `lift_height` 和相对误差快照；Stage5 会记录几乎所有稳定性指标。

***
