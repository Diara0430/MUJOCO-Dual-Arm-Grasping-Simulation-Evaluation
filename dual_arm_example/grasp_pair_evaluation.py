
import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd
import time
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R

import mink
from loop_rate_limiters import RateLimiter

from EnvironmentAndObjects.robot_arm import RobotArm
from EnvironmentAndObjects.scene_object import SceneObject
from utils.common_utils import Format, TrajectoryUtils

# ============================================================
# 0) 配置区：按你自己的 XML / 命名改这里
# ============================================================

XML_PATH = "/home/qianny/Desktop/dual_arm_example/assets/dual_arm_and_single_arm/quad_insert.xml"  # TODO

# 双臂末端 site 名称（你代码里已有）
LEFT_EE_SITE  = "ur_grip_site_left"
RIGHT_EE_SITE = "ur_grip_site_right"

# 关节/actuator 名称（你代码里已有）
ur5_left_arm_joint_names  = ['joint0_ur5left', 'joint1_ur5left', 'joint2_ur5left', 'joint3_ur5left','joint4_ur5left', 'joint5_ur5left']
ur5_right_arm_joint_names = ['joint0_ur5right','joint1_ur5right','joint2_ur5right','joint3_ur5right','joint4_ur5right','joint5_ur5right']
base_joint_name           = ['ur_stand_joint']

ur5_left_arm_first_actuator_name  = 'joint0_motor_ur5left'
ur5_right_arm_first_actuator_name = 'joint0_motor_ur5right'
base_actuator_name                = 'ur_stand_joint_motor'

ur5_left_gripper_actuator_name  = 'gripper_ur5left'
ur5_right_gripper_actuator_name = 'gripper_ur5right'

# 你用于判断接触的夹爪 pad geom 名称（你代码中单臂是 left_finger_pad/right_finger_pad）
# TODO: 改成双臂 UR5 的 finger pad geom 名称（如果你的 xml 里不叫这些）
LEFT_FINGER_PAD_GEOMS  = ["left_pad_ur5left", "right_pad_ur5left"]     # TODO
RIGHT_FINGER_PAD_GEOMS = ["left_pad_ur5right", "right_pad_ur5right"]   # TODO

# 评估参数（符合毕设）
NUM_CANDIDATES = 100
LIFT_HEIGHT = 0.30     # 30 cm
HOLD_TIME = 2.0        # hold 2s
PREGRASP_OFFSET = 0.10 # pre-grasp 距离（沿抓取方向退 10cm）

CONTROL_FREQ = 60.0
SOLVER = "daqp"
MAX_VEL = np.pi

np.random.seed(0)

# ============================================================
# 1) grasp pair 数据结构
# ============================================================

@dataclass
class GraspCandidate:
    left_pos: np.ndarray   # (3,)
    right_pos: np.ndarray  # (3,)
    left_rot: np.ndarray   # (3,3)
    right_rot: np.ndarray  # (3,3)

# ============================================================
# 2) 一些通用工具：构造姿态/采样点/接触检查/打分
# ============================================================

def make_rotation_from_approach(approach_dir, up_hint=np.array([0,0,1.0])):
    """
    给定抓取 approach_dir（末端 z 轴朝向，指向物体），构造一个合理的 3x3 旋转矩阵。
    - z_axis: approach_dir
    - x_axis: up_hint 与 z 的叉乘
    - y_axis: z x x
    """
    z = approach_dir / (np.linalg.norm(approach_dir) + 1e-9)
    x = np.cross(up_hint, z)
    if np.linalg.norm(x) < 1e-6:
        # up_hint 与 z 太平行，换一个 up
        up_hint2 = np.array([0,1.0,0])
        x = np.cross(up_hint2, z)
    x = x / (np.linalg.norm(x) + 1e-9)
    y = np.cross(z, x)
    y = y / (np.linalg.norm(y) + 1e-9)
    rot = np.column_stack([x, y, z])
    return rot

def sample_points_on_bbox(feature_points, n=100):
    """
    从 SceneObject.get_feature_points() 获取的点中，估计 bbox，然后在六个面上随机采样点。
    feature_points: (M,3) M>=8
    返回: n 个点 (n,3)
    """
    pts = np.array(feature_points)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)

    samples = []
    for _ in range(n):
        face = np.random.randint(0, 6)
        u, v = np.random.rand(), np.random.rand()
        x = mins[0] + u*(maxs[0]-mins[0])
        y = mins[1] + v*(maxs[1]-mins[1])
        z = mins[2] + np.random.rand()*(maxs[2]-mins[2])

        p = np.array([x,y,z])

        # 固定某一维到 min/max -> 六个面
        if face == 0: p[0] = mins[0]
        if face == 1: p[0] = maxs[0]
        if face == 2: p[1] = mins[1]
        if face == 3: p[1] = maxs[1]
        if face == 4: p[2] = mins[2]
        if face == 5: p[2] = maxs[2]
        samples.append(p)
    return np.array(samples)

def build_grasp_candidates(scene_obj: SceneObject, num_candidates=100,
                           min_pair_dist=0.08, max_pair_dist=0.40):
    """
    随机生成 grasp pair candidates（符合毕设要求：random sampling ~100）
    简化做法：在物体 bbox 表面采样两点作为左右手抓取点，
    姿态由“指向物体中心”的 approach 方向生成。
    """
    # 用 feature points 估计 bbox
    feature_pts = scene_obj.get_feature_points()
    surface_pts = sample_points_on_bbox(feature_pts, n=num_candidates*5)

    obj_pos, _ = scene_obj.get_xpos_and_rot()
    candidates = []
    trials = 0
    idx = 0

    while len(candidates) < num_candidates and trials < num_candidates*50:
        trials += 1
        p1 = surface_pts[np.random.randint(0, len(surface_pts))]
        p2 = surface_pts[np.random.randint(0, len(surface_pts))]
        dist = np.linalg.norm(p1 - p2)

        if dist < min_pair_dist or dist > max_pair_dist:
            continue

        # 左右手分配策略：按 y 坐标区分（你也可以按 x）
        if p1[1] <= p2[1]:
            left_p, right_p = p1, p2
        else:
            left_p, right_p = p2, p1

        # approach 方向：从抓取点指向物体质心
        left_approach  = (obj_pos - left_p)
        right_approach = (obj_pos - right_p)

        left_rot  = make_rotation_from_approach(left_approach)
        right_rot = make_rotation_from_approach(right_approach)

        candidates.append(GraspCandidate(left_p, right_p, left_rot, right_rot))

    return candidates

def get_geom_ids(model, geom_names):
    ids = []
    for name in geom_names:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid == -1:
            raise ValueError(f"Geom name not found in model: {name}")
        ids.append(gid)
    return ids

def check_contacts(model, data, geom_ids_a, geom_ids_b):
    """
    检查 data.contact 里是否存在 (a in geom_ids_a and b in geom_ids_b) 的接触
    """
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = c.geom1, c.geom2
        if (g1 in geom_ids_a and g2 in geom_ids_b) or (g2 in geom_ids_a and g1 in geom_ids_b):
            return True
    return False

def calculate_slip(T_obj_to_left_init, T_obj_to_right_init, obj_se3, left_se3, right_se3):
    """
    slip：抓住后，T_obj_to_hand 是否变化（变化越大说明滑移/松动）
    返回：左右手 slip 平移量 (m)
    """
    T_obj_to_left_now = obj_se3.inverse().multiply(left_se3)
    T_obj_to_right_now = obj_se3.inverse().multiply(right_se3)

    # Mink SE3：取 translation 差
    dL = np.linalg.norm(T_obj_to_left_now.translation() - T_obj_to_left_init.translation())
    dR = np.linalg.norm(T_obj_to_right_now.translation() - T_obj_to_right_init.translation())
    return dL, dR

# ============================================================
# 3) IK / Task 初始化（复用你现有风格）
# ============================================================

def init_mink(configuration, model, max_velocity, joint_names, ee_sites):
    """
    创建：左右末端 FrameTask + PostureTask + 限制
    ee_sites: [LEFT_EE_SITE, RIGHT_EE_SITE]
    """
    max_velocities = {name: max_velocity for name in joint_names}
    limits = [
        mink.ConfigurationLimit(model=configuration.model),
        mink.VelocityLimit(model, max_velocities)
    ]

    tasks = []
    for site in ee_sites:
        task = mink.FrameTask(
            frame_name=site,
            frame_type="site",
            position_cost=10.0,
            orientation_cost=1.0,
            lm_damping=1.0
        )
        tasks.append(task)

    posture_task = mink.PostureTask(model, cost=1e-3)
    posture_task.set_target(configuration.q)
    tasks.append(posture_task)
    return tasks, limits

# ============================================================
# 4) 评估：执行一个 grasp candidate（核心）
# ============================================================

def evaluate_one_candidate(candidate: GraspCandidate,
                           model, data, viewer, rate,
                           configuration, tasks, limits, solver,
                           ur5_arm_infos, ur5LeftArm, ur5RightArm, obj: SceneObject,
                           finger_geom_ids, object_geom_ids,
                           physics_steps_per_control_step,
                           gripper_open=-1, gripper_close=1):
    """
    返回一个 dict：包含 success/slip/hold_duration 等
    """
    # ---------- 0) 基础读数 ----------
    obj_start_pos, _ = obj.get_xpos_and_rot()
    start_z = obj_start_pos[2]

    # ---------- 1) pre-grasp（沿 approach 方向后退 PREGRASP_OFFSET） ----------
    left_approach = candidate.left_rot[:, 2]   # z axis
    right_approach = candidate.right_rot[:, 2]

    left_pre = candidate.left_pos - PREGRASP_OFFSET * left_approach
    right_pre = candidate.right_pos - PREGRASP_OFFSET * right_approach

    # 构造 SE3 轨迹点（你已有 TrajectoryUtils/Format，但这里直接用 mink）
    def to_se3(pos, rot):
        quat_xyzw = R.from_matrix(rot).as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        return mink.SE3.from_rotation_and_translation(
            rotation=mink.SO3(wxyz=quat_wxyz),
            translation=pos
        )

    # ---------- 2) 移动到 pre-grasp ----------
    # 生成“单步轨迹”：用你已有 execute_trajectory_general 的思想，给它 [[SE3...],[SE3...]]
    traj_pre = [[to_se3(left_pre, candidate.left_rot)], [to_se3(right_pre, candidate.right_rot)]]
    # 让夹爪张开
    # 这里我们用一个短 hold 来让 IK 收敛 + 张开夹爪
    # 你若已有 hold_current_position_and_open_or_close_gripper 可直接用它替换
    for _ in range(int(0.5 / rate.dt)):
        if not viewer.is_running():
            break
        tasks[0].set_target(traj_pre[0][0])
        tasks[1].set_target(traj_pre[1][0])

        configuration.update(data.qpos)
        vel = mink.solve_ik(configuration, tasks, rate.dt, solver, 1e-3, limits=limits)
        configuration.integrate_inplace(vel, rate.dt)

        for i, info in enumerate(ur5_arm_infos):
            c_idx, q_idx, dof = info['ctrl_idx'], info['qpos_idx'], info['range']
            data.ctrl[c_idx:c_idx+dof] = configuration.q[q_idx:q_idx+dof]
            if info['gripper_ctrl_idx'] is not None:
                data.ctrl[info['gripper_ctrl_idx']] = gripper_open

        # physics
        for _p in range(physics_steps_per_control_step):
            for info in ur5_arm_infos:
                c_idx, dof = info['ctrl_idx'], info['range']
                data.qfrc_applied[c_idx:c_idx+dof] = data.qfrc_bias[c_idx:c_idx+dof]
            mujoco.mj_step(model, data)

        viewer.sync()
        rate.sleep()

    # ---------- 3) 从 pre-grasp 进入 grasp 位 ----------
    traj_grasp = [[to_se3(candidate.left_pos, candidate.left_rot)],
                  [to_se3(candidate.right_pos, candidate.right_rot)]]

    for _ in range(int(0.8 / rate.dt)):
        if not viewer.is_running():
            break
        tasks[0].set_target(traj_grasp[0][0])
        tasks[1].set_target(traj_grasp[1][0])

        configuration.update(data.qpos)
        vel = mink.solve_ik(configuration, tasks, rate.dt, solver, 1e-3, limits=limits)
        configuration.integrate_inplace(vel, rate.dt)

        for i, info in enumerate(ur5_arm_infos):
            c_idx, q_idx, dof = info['ctrl_idx'], info['qpos_idx'], info['range']
            data.ctrl[c_idx:c_idx+dof] = configuration.q[q_idx:q_idx+dof]
            if info['gripper_ctrl_idx'] is not None:
                data.ctrl[info['gripper_ctrl_idx']] = gripper_open

        for _p in range(physics_steps_per_control_step):
            for info in ur5_arm_infos:
                c_idx, dof = info['ctrl_idx'], info['range']
                data.qfrc_applied[c_idx:c_idx+dof] = data.qfrc_bias[c_idx:c_idx+dof]
            mujoco.mj_step(model, data)

        viewer.sync()
        rate.sleep()

    # ---------- 4) 关闭夹爪 + 等待稳定 ----------
    for _ in range(int(0.5 / rate.dt)):
        if not viewer.is_running():
            break

        configuration.update(data.qpos)
        vel = mink.solve_ik(configuration, tasks, rate.dt, solver, 1e-3, limits=limits)
        configuration.integrate_inplace(vel, rate.dt)

        for i, info in enumerate(ur5_arm_infos):
            c_idx, q_idx, dof = info['ctrl_idx'], info['qpos_idx'], info['range']
            data.ctrl[c_idx:c_idx+dof] = configuration.q[q_idx:q_idx+dof]
            if info['gripper_ctrl_idx'] is not None:
                data.ctrl[info['gripper_ctrl_idx']] = gripper_close

        for _p in range(physics_steps_per_control_step):
            for info in ur5_arm_infos:
                c_idx, dof = info['ctrl_idx'], info['range']
                data.qfrc_applied[c_idx:c_idx+dof] = data.qfrc_bias[c_idx:c_idx+dof]
            mujoco.mj_step(model, data)

        viewer.sync()
        rate.sleep()

    # 是否有“夹爪 pad 与物体”的接触（粗略判定抓住了）
    has_contact = check_contacts(model, data, finger_geom_ids, object_geom_ids)

    # ---------- 5) 闭链抬升 30cm ----------
    # 抓住后记录相对变换：T_obj_to_left/right（用于后续 slip 计算）
    curr_box_se3 = obj.get_se3()
    curr_left_se3 = ur5LeftArm.get_se3(data)
    curr_right_se3 = ur5RightArm.get_se3(data)

    T_obj_to_left_init = curr_box_se3.inverse().multiply(curr_left_se3)
    T_obj_to_right_init = curr_box_se3.inverse().multiply(curr_right_se3)

    # 物体轨迹：只做竖直上抬（分 60 帧左右）
    steps_lift = int(1.0 / rate.dt)
    lift_traj = []
    for k in range(steps_lift):
        alpha = (k+1) / steps_lift
        target_pos = curr_box_se3.translation().copy()
        target_pos[2] += alpha * LIFT_HEIGHT

        # 维持物体姿态不变（只上抬）
        target_rot = curr_box_se3.rotation().as_matrix()
        quat_xyzw = R.from_matrix(target_rot).as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        lift_traj.append(
            mink.SE3.from_rotation_and_translation(
                rotation=mink.SO3(wxyz=quat_wxyz),
                translation=target_pos
            )
        )

    # 加 RelativeFrameTask 保持双手相对姿态（和你原代码一致）
    T_rel_target = T_obj_to_left_init.inverse().multiply(T_obj_to_right_init)
    task_relative = mink.RelativeFrameTask(
        frame_name=ur5RightArm.attachment_site_name,
        frame_type="site",
        root_name=ur5LeftArm.attachment_site_name,
        root_type="site",
        position_cost=50.0,
        orientation_cost=5.0,
        lm_damping=1.0
    )
    task_relative.set_target(T_rel_target)
    tasks_with_constraint = tasks + [task_relative]

    # 执行抬升：object-centric（把物体轨迹转换为左右手目标）
    def execute_object_centric(object_trajectory, T_obj_to_left, T_obj_to_right):
        for T_world_obj in object_trajectory:
            if not viewer.is_running():
                break
            target_left = T_world_obj.multiply(T_obj_to_left)
            target_right = T_world_obj.multiply(T_obj_to_right)

            tasks_with_constraint[0].set_target(target_left)
            tasks_with_constraint[1].set_target(target_right)

            configuration.update(data.qpos)
            vel = mink.solve_ik(configuration, tasks_with_constraint, rate.dt, solver, 1e-3, limits=limits)
            configuration.integrate_inplace(vel, rate.dt)

            for i, info in enumerate(ur5_arm_infos):
                c_idx, q_idx, dof = info['ctrl_idx'], info['qpos_idx'], info['range']
                data.ctrl[c_idx:c_idx+dof] = configuration.q[q_idx:q_idx+dof]
                if info['gripper_ctrl_idx'] is not None:
                    data.ctrl[info['gripper_ctrl_idx']] = gripper_close

            for _p in range(physics_steps_per_control_step):
                for info in ur5_arm_infos:
                    c_idx, dof = info['ctrl_idx'], info['range']
                    data.qfrc_applied[c_idx:c_idx+dof] = data.qfrc_bias[c_idx:c_idx+dof]
                mujoco.mj_step(model, data)

            viewer.sync()
            rate.sleep()

    execute_object_centric(lift_traj, T_obj_to_left_init, T_obj_to_right_init)

    # ---------- 6) hold 并测 hold_duration + slip ----------
    hold_steps = int(HOLD_TIME / rate.dt)
    hold_duration = 0.0
    max_slip_L = 0.0
    max_slip_R = 0.0

    for t in range(hold_steps):
        if not viewer.is_running():
            break

        # 保持最后一个 lift 目标
        T_world_obj_target = lift_traj[-1]
        target_left = T_world_obj_target.multiply(T_obj_to_left_init)
        target_right = T_world_obj_target.multiply(T_obj_to_right_init)
        tasks_with_constraint[0].set_target(target_left)
        tasks_with_constraint[1].set_target(target_right)

        configuration.update(data.qpos)
        vel = mink.solve_ik(configuration, tasks_with_constraint, rate.dt, solver, 1e-3, limits=limits)
        configuration.integrate_inplace(vel, rate.dt)

        for i, info in enumerate(ur5_arm_infos):
            c_idx, q_idx, dof = info['ctrl_idx'], info['qpos_idx'], info['range']
            data.ctrl[c_idx:c_idx+dof] = configuration.q[q_idx:q_idx+dof]
            if info['gripper_ctrl_idx'] is not None:
                data.ctrl[info['gripper_ctrl_idx']] = gripper_close

        for _p in range(physics_steps_per_control_step):
            for info in ur5_arm_infos:
                c_idx, dof = info['ctrl_idx'], info['range']
                data.qfrc_applied[c_idx:c_idx+dof] = data.qfrc_bias[c_idx:c_idx+dof]
            mujoco.mj_step(model, data)

        # 计算 slip
        curr_box_se3 = obj.get_se3()
        curr_left_se3 = ur5LeftArm.get_se3(data)
        curr_right_se3 = ur5RightArm.get_se3(data)
        dL, dR = calculate_slip(T_obj_to_left_init, T_obj_to_right_init, curr_box_se3, curr_left_se3, curr_right_se3)
        max_slip_L = max(max_slip_L, dL)
        max_slip_R = max(max_slip_R, dR)

        # 是否还接触（粗略判定没掉）
        still_contact = check_contacts(model, data, finger_geom_ids, object_geom_ids)

        # 物体是否掉落（z 低于某阈值）
        obj_pos_now, _ = obj.get_xpos_and_rot()
        if (not still_contact) and obj_pos_now[2] < start_z + 0.02:
            break

        hold_duration += rate.dt
        viewer.sync()
        rate.sleep()

    # ---------- 7) 成功判定 ----------
    obj_end_pos, _ = obj.get_xpos_and_rot()
    lifted = (obj_end_pos[2] - start_z) > 0.10  # 抬起来至少 10cm 认为 lift 成功
    success = int(lifted and has_contact)

    return {
        "success": success,
        "has_contact": int(has_contact),
        "final_lifted_height": float(obj_end_pos[2] - start_z),
        "hold_duration": float(hold_duration),
        "max_slip_left": float(max_slip_L),
        "max_slip_right": float(max_slip_R),
        "max_slip": float(max(max_slip_L, max_slip_R))
    }

# ============================================================
# 5) 主程序：对每个物体采样100个candidate并评估
# ============================================================

def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    configuration = mink.Configuration(model)

    sim_dt = model.opt.timestep
    control_dt = 1.0 / CONTROL_FREQ
    physics_steps_per_control_step = int(np.ceil(control_dt / sim_dt))

    # 初始化 UR5 双臂
    ur5LeftArm = RobotArm('ur5_left', ur5_left_arm_joint_names, LEFT_EE_SITE,
                          ur5_left_arm_first_actuator_name, ur5_left_gripper_actuator_name)
    ur5RightArm = RobotArm('ur5_right', ur5_right_arm_joint_names, RIGHT_EE_SITE,
                           ur5_right_arm_first_actuator_name, ur5_right_gripper_actuator_name)
    ur5BaseBody = RobotArm('base', base_joint_name, None, base_actuator_name)

    ur5LeftArm.initialize(model)
    ur5RightArm.initialize(model)
    ur5BaseBody.initialize(model)

    ur5_arm_infos = [ur5BaseBody.get_control_info(),
                     ur5LeftArm.get_control_info(),
                     ur5RightArm.get_control_info()]

    # 任务初始化（左右末端 task）
    ur5_all_joints_names = base_joint_name + ur5_left_arm_joint_names + ur5_right_arm_joint_names
    tasks, limits = init_mink(configuration, model, MAX_VEL, ur5_all_joints_names, [LEFT_EE_SITE, RIGHT_EE_SITE])

    # 准备接触检测用 geom id
    finger_geom_ids = get_geom_ids(model, LEFT_FINGER_PAD_GEOMS + RIGHT_FINGER_PAD_GEOMS)

    # 物体列表：你可以扩展多个 object（毕设要求：diverse objects）
    # TODO：改成你 xml 里实际存在的 body/geom 名称；或你后面导入 YCB 替换
    object_specs = [
        {"body": "big_box", "geom": "big_box_geom"},
        {"body": "small_box", "geom": "small_box_geom"},
    ]

    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        viewer.cam.lookat[:] = [1.0, 0.0, 0.5]
        viewer.cam.azimuth = -90
        viewer.cam.elevation = -15
        viewer.cam.distance = 5.0

        rate = RateLimiter(frequency=CONTROL_FREQ, warn=False)
        viewer.sync()

        all_results = []

        # 先“预热”一下让物体落稳
        for _ in range(500):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)

        # 保存一个初始状态，用于每个 candidate 复位（关键！）
        qpos_init = data.qpos.copy()
        qvel_init = data.qvel.copy()

        for obj_spec in object_specs:
            obj = SceneObject(obj_spec["body"], model, data, obj_spec["geom"])
            object_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, obj_spec["geom"])
            if object_geom_id == -1:
                raise ValueError(f"Object geom not found: {obj_spec['geom']}")
            object_geom_ids = [object_geom_id]

            print(f"\n========== Evaluating object: {obj_spec['body']} ==========")

            # 生成 candidates
            candidates = build_grasp_candidates(obj, num_candidates=NUM_CANDIDATES)
            print(f"Generated {len(candidates)} grasp pair candidates.")

            for i, cand in enumerate(candidates):
                # ---- 每个 candidate 都要 reset 仿真（非常重要）----
                data.qpos[:] = qpos_init
                data.qvel[:] = qvel_init
                mujoco.mj_forward(model, data)

                # 评估
                try:
                    metrics = evaluate_one_candidate(
                        cand, model, data, viewer, rate,
                        configuration, tasks, limits, SOLVER,
                        ur5_arm_infos, ur5LeftArm, ur5RightArm, obj,
                        finger_geom_ids, object_geom_ids,
                        physics_steps_per_control_step
                    )
                except Exception as e:
                    metrics = {"success": 0, "has_contact": 0,
                               "final_lifted_height": 0.0, "hold_duration": 0.0,
                               "max_slip_left": 999, "max_slip_right": 999, "max_slip": 999}
                    print(f"[{obj_spec['body']}] candidate {i} failed due to exception: {e}")

                # 记录 candidate 的几何特征（用于结果分析/画图）
                pair_dist = float(np.linalg.norm(cand.left_pos - cand.right_pos))
                row = {
                    "object": obj_spec["body"],
                    "candidate_id": i,
                    "pair_dist": pair_dist,
                    "left_pos_x": cand.left_pos[0], "left_pos_y": cand.left_pos[1], "left_pos_z": cand.left_pos[2],
                    "right_pos_x": cand.right_pos[0], "right_pos_y": cand.right_pos[1], "right_pos_z": cand.right_pos[2],
                }
                row.update(metrics)
                all_results.append(row)

                if (i+1) % 10 == 0:
                    print(f"  Progress: {i+1}/{len(candidates)} | success so far: {sum(r['success'] for r in all_results if r['object']==obj_spec['body'])}")

        # 汇总结果保存
        df = pd.DataFrame(all_results)
        df.to_csv("grasp_pair_eval_results.csv", index=False)
        print("\nSaved results to grasp_pair_eval_results.csv")

        # 输出每个物体 top-5（按 success->hold_duration->slip 排序）
        for obj_name in df["object"].unique():
            sub = df[df["object"] == obj_name].copy()
            sub = sub.sort_values(by=["success", "hold_duration", "max_slip"],
                                  ascending=[False, False, True])
            print(f"\nTop-5 candidates for {obj_name}:")
            print(sub.head(5)[["candidate_id","success","final_lifted_height","hold_duration","max_slip","pair_dist"]])

        # viewer 不会自动退出，这里停住
        while viewer.is_running():
            viewer.sync()
            rate.sleep()

if __name__ == "__main__":
    main()