
import time
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R

import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter
import mink

from EnvironmentAndObjects.robot_arm import RobotArm
from EnvironmentAndObjects.scene_object import SceneObject


# ============================================================
# 0) 配置区：改这里就能跑
# ============================================================

XML_PATH = "/home/qianny/Desktop/dual_arm_example/assets/dual_arm_and_single_arm/quad_insert.xml"

# 双臂末端 site（来自 dual_ur5.xml）
LEFT_EE_SITE  = "ur_grip_site_left"
RIGHT_EE_SITE = "ur_grip_site_right"

# base + joint names（你现成）
base_joint_name           = ['ur_stand_joint']
ur5_left_arm_joint_names  = ['joint0_ur5left', 'joint1_ur5left', 'joint2_ur5left', 'joint3_ur5left',
                             'joint4_ur5left', 'joint5_ur5left']
ur5_right_arm_joint_names = ['joint0_ur5right', 'joint1_ur5right', 'joint2_ur5right', 'joint3_ur5right',
                             'joint4_ur5right', 'joint5_ur5right']

base_actuator_name                = 'ur_stand_joint_motor'
ur5_left_arm_first_actuator_name  = 'joint0_motor_ur5left'
ur5_right_arm_first_actuator_name = 'joint0_motor_ur5right'
ur5_left_gripper_actuator_name    = 'gripper_ur5left'
ur5_right_gripper_actuator_name   = 'gripper_ur5right'

# ✅ 夹爪“真正用于接触”的指尖 box geom（来自 dual_ur5.xml）
LEFT_FINGERTIP_GEOMS = ["left_fingertip_visual_ur5left", "right_fingertip_visual_ur5left"]
RIGHT_FINGERTIP_GEOMS = ["left_fingertip_visual_ur5right", "right_fingertip_visual_ur5right"]

# world.xml 里的物体（你可以继续加）
OBJECT_SPECS = [
    {"body": "big_box", "geom": "big_box_geom"},
    {"body": "small_box", "geom": "small_box_geom"},
]

# 采样与评估参数（毕设要求）
NUM_CANDIDATES = 100
PREGRASP_OFFSET = 0.10   # 预抓取退后 10cm
LIFT_HEIGHT = 0.30       # 抬升 30cm
HOLD_TIME = 2.0          # 保持 2s

# 控制参数
CONTROL_FREQ = 60.0
DT = 1.0 / CONTROL_FREQ
SOLVER = "daqp"
MAX_VEL = np.pi

# 夹爪控制（你原代码用 [-1,1]，但注意 motor 的方向可能需要你微调）
GRIPPER_OPEN = -1.0
GRIPPER_CLOSE =  1.0

np.random.seed(0)


# ============================================================
# 1) 数据结构：抓取候选对
# ============================================================

@dataclass
class GraspCandidate:
    left_pos: np.ndarray   # (3,)
    right_pos: np.ndarray  # (3,)
    left_rot: np.ndarray   # (3,3)
    right_rot: np.ndarray  # (3,3)


# ============================================================
# 2) 工具函数：姿态构造 / AABB 采样 / 接触检测 / slip 计算
# ============================================================

def make_rotation_from_approach(approach_dir, up_hint=np.array([0, 0, 1.0])):
    """
    给定抓取 approach_dir（末端 z 轴朝向），构造一个正交旋转矩阵:
      z = approach_dir
      x = up_hint × z
      y = z × x
    """
    z = approach_dir / (np.linalg.norm(approach_dir) + 1e-9)
    x = np.cross(up_hint, z)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(np.array([0, 1.0, 0]), z)
    x = x / (np.linalg.norm(x) + 1e-9)
    y = np.cross(z, x)
    y = y / (np.linalg.norm(y) + 1e-9)
    return np.column_stack([x, y, z])


def sample_points_on_aabb(feature_points, n=200):
    """
    用 feature_points 估计 AABB，在六个面上随机采样点。
    适用于 box / 规则物体（你当前 world.xml 的两个 box 非常适合）。
    """
    pts = np.asarray(feature_points)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)

    samples = []
    for _ in range(n):
        face = np.random.randint(0, 6)
        u, v = np.random.rand(), np.random.rand()
        p = np.array([
            mins[0] + u * (maxs[0] - mins[0]),
            mins[1] + v * (maxs[1] - mins[1]),
            mins[2] + np.random.rand() * (maxs[2] - mins[2]),
        ])
        if face == 0: p[0] = mins[0]
        if face == 1: p[0] = maxs[0]
        if face == 2: p[1] = mins[1]
        if face == 3: p[1] = maxs[1]
        if face == 4: p[2] = mins[2]
        if face == 5: p[2] = maxs[2]
        samples.append(p)
    return np.array(samples)


def build_grasp_candidates(obj: SceneObject, num_candidates=100,
                           min_pair_dist=0.10, max_pair_dist=0.50):
    """
    随机采样 grasp pair candidates（符合毕设 Method 2）。
    简化：AABB 面上采样两点，左右手姿态都“朝向物体质心”。
    """
    feature_pts = obj.get_feature_points()
    surface_pts = sample_points_on_aabb(feature_pts, n=num_candidates * 10)

    obj_pos, _ = obj.get_xpos_and_rot()

    candidates = []
    tries = 0
    while len(candidates) < num_candidates and tries < num_candidates * 200:
        tries += 1
        p1 = surface_pts[np.random.randint(len(surface_pts))]
        p2 = surface_pts[np.random.randint(len(surface_pts))]
        d = np.linalg.norm(p1 - p2)
        if d < min_pair_dist or d > max_pair_dist:
            continue

        # 左右手分配：按 y 值（也可以按 x）
        if p1[1] <= p2[1]:
            left_p, right_p = p1, p2
        else:
            left_p, right_p = p2, p1

        left_rot = make_rotation_from_approach(obj_pos - left_p)
        right_rot = make_rotation_from_approach(obj_pos - right_p)

        candidates.append(GraspCandidate(left_p, right_p, left_rot, right_rot))

    return candidates


def to_mink_se3(pos, rot):
    """pos(3,), rot(3,3) -> mink.SE3"""
    quat_xyzw = R.from_matrix(rot).as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    return mink.SE3.from_rotation_and_translation(
        rotation=mink.SO3(wxyz=quat_wxyz),
        translation=pos
    )


def get_geom_ids(model, geom_names):
    ids = []
    for name in geom_names:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid == -1:
            raise ValueError(f"Geom not found: {name}")
        ids.append(gid)
    return ids


def contact_exists(model, data, geom_ids_a, geom_ids_b):
    """检查接触对是否存在，用于判断“是否抓住/是否仍接触”"""
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = c.geom1, c.geom2
        if (g1 in geom_ids_a and g2 in geom_ids_b) or (g2 in geom_ids_a and g1 in geom_ids_b):
            return True
    return False


def calc_slip(obj_se3, left_se3, right_se3, T_obj_to_left_init, T_obj_to_right_init):
    """
    slip 指标：抓住后，物体坐标系到手坐标系的相对平移是否变化。
    """
    T_obj_to_left_now = obj_se3.inverse().multiply(left_se3)
    T_obj_to_right_now = obj_se3.inverse().multiply(right_se3)

    dL = np.linalg.norm(T_obj_to_left_now.translation() - T_obj_to_left_init.translation())
    dR = np.linalg.norm(T_obj_to_right_now.translation() - T_obj_to_right_init.translation())
    return dL, dR


# ============================================================
# 3) Mink Task 初始化
# ============================================================

def init_tasks(configuration, model, joint_names, ee_sites, max_vel=MAX_VEL):
    """
    tasks = [left FrameTask, right FrameTask, PostureTask]
    limits = [ConfigurationLimit, VelocityLimit]
    """
    max_velocities = {name: max_vel for name in joint_names}
    limits = [
        mink.ConfigurationLimit(model=configuration.model),
        mink.VelocityLimit(model, max_velocities),
    ]

    tasks = []
    for site in ee_sites:
        tasks.append(mink.FrameTask(
            frame_name=site, frame_type="site",
            position_cost=10.0, orientation_cost=1.0,
            lm_damping=1.0
        ))

    posture = mink.PostureTask(model, cost=1e-3)
    posture.set_target(configuration.q)
    tasks.append(posture)
    return tasks, limits


# ============================================================
# 4) 评估一个 candidate（毕设 Method 3 核心）
# ============================================================

def step_ik_and_sim(model, data, configuration, tasks, limits, solver, rate,
                    arm_infos, gripper_val, physics_steps):
    """
    做一次：设置好 tasks target 后 -> solve IK -> 写 ctrl -> 多步 mj_step
    """
    configuration.update(data.qpos)
    vel = mink.solve_ik(configuration, tasks, rate.dt, solver, 1e-3, limits=limits)
    configuration.integrate_inplace(vel, rate.dt)

    # 写控制
    for info in arm_infos:
        c_idx, q_idx, dof = info['ctrl_idx'], info['qpos_idx'], info['range']
        data.ctrl[c_idx:c_idx+dof] = configuration.q[q_idx:q_idx+dof]
        if info.get('gripper_ctrl_idx', None) is not None:
            data.ctrl[info['gripper_ctrl_idx']] = gripper_val

    # 多个 physics step
    for _ in range(physics_steps):
        for info in arm_infos:
            c_idx, dof = info['ctrl_idx'], info['range']
            data.qfrc_applied[c_idx:c_idx+dof] = data.qfrc_bias[c_idx:c_idx+dof]  # 重力补偿
        mujoco.mj_step(model, data)

    rate.sleep()


def evaluate_candidate(candidate: GraspCandidate,
                       model, data, configuration, tasks, limits, solver,
                       viewer, rate, physics_steps,
                       ur5LeftArm, ur5RightArm, arm_infos,
                       obj: SceneObject,
                       fingertip_geom_ids, object_geom_ids):
    


    """
    输出：success / hold_duration / max_slip / lift_height 等
    """

    # ---------- 0) 初始物体高度 ----------
    obj_pos0, _ = obj.get_xpos_and_rot()
    z0 = obj_pos0[2]

    # ---------- 1) pre-grasp ----------
    left_approach = candidate.left_rot[:, 2]
    right_approach = candidate.right_rot[:, 2]
    left_pre = candidate.left_pos - PREGRASP_OFFSET * left_approach
    right_pre = candidate.right_pos - PREGRASP_OFFSET * right_approach

    T_left_pre = to_mink_se3(left_pre, candidate.left_rot)
    T_right_pre = to_mink_se3(right_pre, candidate.right_rot)

    # 让夹爪张开并收敛到 pre-grasp
    for _ in range(int(0.6 / rate.dt)):
        if not viewer.is_running():
            break
        tasks[0].set_target(T_left_pre)
        tasks[1].set_target(T_right_pre)
        step_ik_and_sim(model, data, configuration, tasks, limits, solver, rate,
                        arm_infos, GRIPPER_OPEN, physics_steps)
        viewer.sync()

    # ---------- 2) 进入 grasp pose ----------
    T_left_grasp = to_mink_se3(candidate.left_pos, candidate.left_rot)
    T_right_grasp = to_mink_se3(candidate.right_pos, candidate.right_rot)

    for _ in range(int(0.8 / rate.dt)):
        if not viewer.is_running():
            break
        tasks[0].set_target(T_left_grasp)
        tasks[1].set_target(T_right_grasp)
        step_ik_and_sim(model, data, configuration, tasks, limits, solver, rate,
                        arm_infos, GRIPPER_OPEN, physics_steps)
        viewer.sync()

    # ---------- 3) 闭合夹爪 ----------
    for _ in range(int(0.5 / rate.dt)):
        if not viewer.is_running():
            break
        # 保持 grasp target
        tasks[0].set_target(T_left_grasp)
        tasks[1].set_target(T_right_grasp)
        step_ik_and_sim(model, data, configuration, tasks, limits, solver, rate,
                        arm_infos, GRIPPER_CLOSE, physics_steps)
        viewer.sync()

    # 是否发生了“指尖-物体”接触（抓住的必要条件之一）
    has_contact = contact_exists(model, data, fingertip_geom_ids, object_geom_ids)

    # ---------- 4) 建立闭链约束并抬升 30cm ----------
    # 记录当前物体与双手的相对变换（用于 slip）
    obj_se3 = obj.get_se3()
    left_se3 = ur5LeftArm.get_se3(data)
    right_se3 = ur5RightArm.get_se3(data)

    T_obj_to_left_init = obj_se3.inverse().multiply(left_se3)
    T_obj_to_right_init = obj_se3.inverse().multiply(right_se3)

    # 相对约束：保持左右手相对位姿固定
    T_rel_target = T_obj_to_left_init.inverse().multiply(T_obj_to_right_init)
    task_relative = mink.RelativeFrameTask(
        frame_name=ur5RightArm.attachment_site_name, frame_type="site",
        root_name=ur5LeftArm.attachment_site_name, root_type="site",
        position_cost=50.0, orientation_cost=5.0, lm_damping=1.0
    )
    task_relative.set_target(T_rel_target)

    tasks_closed_chain = tasks + [task_relative]

    # 物体目标轨迹：只上抬（保持姿态不变）
    steps_lift = int(1.0 / rate.dt)
    obj_rot = obj_se3.rotation().as_matrix()
    obj_pos = obj_se3.translation().copy()

    for k in range(steps_lift):
        if not viewer.is_running():
            break
        alpha = (k + 1) / steps_lift
        target_pos = obj_pos.copy()
        target_pos[2] += alpha * LIFT_HEIGHT
        T_world_obj = to_mink_se3(target_pos, obj_rot)

        # object-centric：根据物体目标反算双手目标
        target_left = T_world_obj.multiply(T_obj_to_left_init)
        target_right = T_world_obj.multiply(T_obj_to_right_init)

        tasks_closed_chain[0].set_target(target_left)
        tasks_closed_chain[1].set_target(target_right)

        # IK + sim（夹爪保持闭合）
        step_ik_and_sim(model, data, configuration, tasks_closed_chain, limits, solver, rate,
                        arm_infos, GRIPPER_CLOSE, physics_steps)
        viewer.sync()

    # 抬升后高度
    obj_pos1, _ = obj.get_xpos_and_rot()
    lift_height = float(obj_pos1[2] - z0)

    # ---------- 5) hold 阶段：统计 hold_duration + slip ----------
    hold_steps = int(HOLD_TIME / rate.dt)
    hold_duration = 0.0
    max_slip = 0.0

    # hold 的目标：保持最后一个抬升目标
    T_world_obj_final = to_mink_se3(obj_pos + np.array([0, 0, LIFT_HEIGHT]), obj_rot)

    for _ in range(hold_steps):
        if not viewer.is_running():
            break

        target_left = T_world_obj_final.multiply(T_obj_to_left_init)
        target_right = T_world_obj_final.multiply(T_obj_to_right_init)
        tasks_closed_chain[0].set_target(target_left)
        tasks_closed_chain[1].set_target(target_right)

        step_ik_and_sim(model, data, configuration, tasks_closed_chain, limits, solver, rate,
                        arm_infos, GRIPPER_CLOSE, physics_steps)
        viewer.sync()

        # slip
        obj_se3_now = obj.get_se3()
        left_se3_now = ur5LeftArm.get_se3(data)
        right_se3_now = ur5RightArm.get_se3(data)
        dL, dR = calc_slip(obj_se3_now, left_se3_now, right_se3_now, T_obj_to_left_init, T_obj_to_right_init)
        max_slip = max(max_slip, dL, dR)

        # 掉落判定（两种：失去接触 + 高度下降）
        still_contact = contact_exists(model, data, fingertip_geom_ids, object_geom_ids)
        obj_p, _ = obj.get_xpos_and_rot()
        if (not still_contact) and (obj_p[2] < z0 + 0.02):
            break

        hold_duration += rate.dt

    # ---------- 6) success 判定 ----------
    # 毕设简单定义：抬升超过 10cm 且抓取时确实接触过
    success = int((lift_height > 0.10) and has_contact)

    return {
        "success": success,
        "has_contact": int(has_contact),
        "lift_height": lift_height,
        "hold_duration": float(hold_duration),
        "max_slip": float(max_slip),
    }


# ============================================================
# 6) Reset 机制：保存并恢复初始状态（用于批量评估）
# ============================================================

def save_state(model, data):
    """
    更稳的 full-state 保存/恢复（如果你 mujoco 版本支持 mj_getState / mj_setState）。
    如果不支持，也至少返回 qpos/qvel。
    """
    try:
        sz = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        buf = np.zeros(sz, dtype=np.float64)
        mujoco.mj_getState(model, data, buf, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        return ("full", buf)
    except Exception:
        return ("simple", data.qpos.copy(), data.qvel.copy())


def load_state(model, data, state):
    if state[0] == "full":
        buf = state[1]
        mujoco.mj_setState(model, data, buf, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    else:
        _, qpos, qvel = state
        data.qpos[:] = qpos
        data.qvel[:] = qvel
    mujoco.mj_forward(model, data)


# ============================================================
# 7) 主程序：每个物体采样100对 candidate，并评估写 CSV
# ============================================================

def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    configuration = mink.Configuration(model)

    sim_dt = model.opt.timestep
    physics_steps = int(np.ceil(DT / sim_dt))

    # 初始化双臂对象
    ur5LeftArm = RobotArm('ur5_left', ur5_left_arm_joint_names, LEFT_EE_SITE,
                          ur5_left_arm_first_actuator_name, ur5_left_gripper_actuator_name)
    ur5RightArm = RobotArm('ur5_right', ur5_right_arm_joint_names, RIGHT_EE_SITE,
                           ur5_right_arm_first_actuator_name, ur5_right_gripper_actuator_name)
    ur5BaseBody = RobotArm('base', base_joint_name, None, base_actuator_name)

    ur5LeftArm.initialize(model)
    ur5RightArm.initialize(model)
    ur5BaseBody.initialize(model)

    # 控制信息（你的 RobotArm.get_control_info() 应该返回 ctrl_idx/qpos_idx/range/gripper_ctrl_idx）
    arm_infos = [ur5BaseBody.get_control_info(),
                 ur5LeftArm.get_control_info(),
                 ur5RightArm.get_control_info()]

    # Mink tasks
    all_joint_names = base_joint_name + ur5_left_arm_joint_names + ur5_right_arm_joint_names
    tasks, limits = init_tasks(configuration, model, all_joint_names, [LEFT_EE_SITE, RIGHT_EE_SITE], MAX_VEL)

    # 接触检测 geom id
    fingertip_geom_ids = get_geom_ids(model, LEFT_FINGERTIP_GEOMS + RIGHT_FINGERTIP_GEOMS)

    # viewer + rate
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        rate = RateLimiter(frequency=CONTROL_FREQ, warn=False)

        # 预热：让物体自然落稳
        for _ in range(500):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)

        # 保存初始状态（用于每个 candidate reset）
        init_state = save_state(model, data)

        results = []

        for spec in OBJECT_SPECS:
            obj = SceneObject(spec["body"], model, data, spec["geom"])

            obj_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, spec["geom"])
            if obj_geom_id == -1:
                raise ValueError(f"Object geom not found: {spec['geom']}")
            object_geom_ids = [obj_geom_id]

            print(f"\n========== Object: {spec['body']} ==========")

            # 生成 100 对 grasp candidates（毕设 Method 2）
            candidates = build_grasp_candidates(obj, NUM_CANDIDATES)
            print(f"Generated {len(candidates)} candidates.")

            for i, cand in enumerate(candidates):
                if not viewer.is_running():
                    break

                # reset
                load_state(model, data, init_state)

                # 评估（毕设 Method 3）
                try:
                    metrics = evaluate_candidate(
                        cand, model, data, configuration, tasks, limits, SOLVER,
                        viewer, rate, physics_steps,
                        ur5LeftArm, ur5RightArm, arm_infos,
                        obj, fingertip_geom_ids, object_geom_ids
                    )
                except Exception as e:
                    print(f"[{spec['body']}] candidate {i} exception: {e}")
                    metrics = {"success": 0, "has_contact": 0, "lift_height": 0.0, "hold_duration": 0.0, "max_slip": 999.0}

                # 记录用于统计的几何特征（后面写报告很关键）
                pair_dist = float(np.linalg.norm(cand.left_pos - cand.right_pos))
                row = {
                    "object": spec["body"],
                    "candidate_id": i,
                    "pair_dist": pair_dist,
                    "left_x": cand.left_pos[0], "left_y": cand.left_pos[1], "left_z": cand.left_pos[2],
                    "right_x": cand.right_pos[0], "right_y": cand.right_pos[1], "right_z": cand.right_pos[2],
                }
                row.update(metrics)
                results.append(row)

                if (i + 1) % 10 == 0:
                    succ = sum(r["success"] for r in results if r["object"] == spec["body"])
                    print(f"  progress {i+1}/{len(candidates)} | success so far: {succ}")

        # 保存 CSV（毕设 Method 4 的数据来源）
        df = pd.DataFrame(results)
        df.to_csv("grasp_pair_eval_results.csv", index=False)
        print("\nSaved: grasp_pair_eval_results.csv")

        # 打印每个物体 Top-5 grasp pair
        for obj_name in df["object"].unique():
            sub = df[df["object"] == obj_name].copy()
            # 优先 success，其次 hold_duration 大，其次 slip 小
            sub = sub.sort_values(by=["success", "hold_duration", "max_slip"],
                                  ascending=[False, False, True])
            print(f"\nTop-5 for {obj_name}:")
            print(sub.head(5)[["candidate_id", "success", "lift_height", "hold_duration", "max_slip", "pair_dist"]])

        # 保持窗口
        while viewer.is_running():
            viewer.sync()
            rate.sleep()


if __name__ == "__main__":
    main()
