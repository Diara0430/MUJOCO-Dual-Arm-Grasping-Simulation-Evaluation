import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter
import time

from EnvironmentAndObjects.robot_arm import RobotArm
from utils.common_utils import Format, TrajectoryUtils
from EnvironmentAndObjects.scene_object import SceneObject
from utils.promp_utils import SimpleProMP

import mink
from scipy.spatial.transform import Rotation as R, Slerp

xml_path = "/home/qianny/Desktop/dual_arm_example/assets/dual_arm_and_single_arm/quad_insert.xml"


def apply_local_rotation(curr_rot_mat, axis='z', angle_deg=90):
    r_curr = R.from_matrix(curr_rot_mat)
    r_delta = R.from_euler(axis, angle_deg, degrees=True)
    r_target = r_curr * r_delta
    return r_target.as_matrix()


# ==========================================
# 人工轨迹生成 (标准化版)
# ==========================================

def generate_multi_cartesian_trajectories(start_pos_list, end_pos_list,
                                          start_rot_list, end_rot_list,
                                          linear_speed,
                                          control_dt,
                                          peak_height=0.0):
    """
    输入不管是一个点还是一组点，不管是list还是numpy，
    统一都会被处理成多臂模式。
    """
    # 1. 强制转为 (N, 3) 和 (N, 3, 3)
    start_pos_arr = Format.ensure_numpy_2d(start_pos_list)
    end_pos_arr = Format.ensure_numpy_2d(end_pos_list)
    start_rot_arr = Format.ensure_rotation_batch(start_rot_list)
    end_rot_arr = Format.ensure_rotation_batch(end_rot_list)

    all_trajectories = []

    # zip 遍历第一维 (N)
    iterator = zip(start_pos_arr, end_pos_arr, start_rot_arr, end_rot_arr)

    for i, (start_pos, end_pos, start_rotation_matrix, end_rotation_matrix) in enumerate(iterator):
        displacement = end_pos - start_pos
        distance = np.linalg.norm(displacement)

        if distance < 1e-6:
            # 静止情况
            start_rot = R.from_matrix(start_rotation_matrix)
            x, y, z, w = start_rot.as_quat()
            quat_wxyz = np.array([w, x, y, z])
            static_se3 = mink.SE3.from_rotation_and_translation(
                rotation=mink.SO3(wxyz=quat_wxyz),
                translation=start_pos
            )
            all_trajectories.append([static_se3])
            continue

        total_time = distance / linear_speed
        num_points = max(2, int(np.ceil(total_time / control_dt)))

        # 位置插值
        positions = np.linspace(start_pos, end_pos, num_points)
        if peak_height > 0:
            t_steps = np.linspace(0, 1, num_points)
            z_offsets = peak_height * np.sin(t_steps * np.pi)
            positions[:, 2] += z_offsets

        # 旋转插值
        start_rot_obj = R.from_matrix(start_rotation_matrix)
        end_rot_obj = R.from_matrix(end_rotation_matrix)
        key_times = [0.0, 1.0]
        key_rots = R.concatenate([start_rot_obj, end_rot_obj])
        slerp = Slerp(key_times, key_rots)
        interp_times = np.linspace(0, 1, num_points)
        interp_rots = slerp(interp_times)
        interp_quats_xyzw = interp_rots.as_quat()

        se3_trajectory = []
        for j in range(num_points):
            pos = positions[j]
            x, y, z, w = interp_quats_xyzw[j]
            quat_wxyz = np.array([w, x, y, z])
            se3 = mink.SE3.from_rotation_and_translation(
                rotation=mink.SO3(wxyz=quat_wxyz),
                translation=pos
            )
            se3_trajectory.append(se3)

        all_trajectories.append(se3_trajectory)

    return all_trajectories


def generate_rotation_trajectory(fixed_pos, start_rot_matrix, end_rot_matrix, duration, control_dt):
    """辅助函数：单臂纯旋转"""
    num_points = max(2, int(duration / control_dt))
    key_times = [0.0, 1.0]
    key_rots = R.from_matrix([start_rot_matrix, end_rot_matrix])
    slerp = Slerp(key_times, key_rots)
    interp_times = np.linspace(0, 1, num_points)
    interp_rots = slerp(interp_times)

    trajectory = []
    for quat in interp_rots.as_quat():
        quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])
        se3 = mink.SE3.from_rotation_and_translation(
            rotation=mink.SO3(wxyz=quat_wxyz),
            translation=fixed_pos
        )
        trajectory.append(se3)
    return trajectory


def plan_multi_gripper_rotation(data, arm_specs, global_duration=1.0, dt=0.002):
    """批量规划旋转，确保返回 List[List[SE3]]"""
    trajectories = []
    # 确保 arm_specs 是列表
    if not isinstance(arm_specs, list): arm_specs = [arm_specs]

    for spec in arm_specs:
        site_id = spec['site_id']
        angle = spec['angle']
        axis = spec.get('axis', 'z')
        duration = spec.get('duration', global_duration)

        curr_pos = data.site_xpos[site_id].copy()
        curr_rot_mat = data.site_xmat[site_id].copy().reshape(3, 3)

        r_curr = R.from_matrix(curr_rot_mat)
        r_delta = R.from_euler(axis, angle, degrees=True)
        target_rot_mat = (r_curr * r_delta).as_matrix()

        traj = generate_rotation_trajectory(curr_pos, curr_rot_mat, target_rot_mat, duration, dt)
        trajectories.append(traj)

    return trajectories


def generate_big_box_trajectory_promp(small_box_data_input, big_start_pose, big_end_pos,
                                      single_arm_base_pose=np.array([2, 0, 0]),
                                      dual_arm_robot_base_pos=np.array([0, 0, 0]),
                                      n_basis=20, rot_scale=0.5):
    """
        参考轨迹list传入后，我们对齐里面的每条轨迹到x轴正向，然后提取里面的纯轨迹形状，传入promp
        我们再提取轨迹里面的旋转信息，传入promp
        分别生成从新起点到新终点的transaction 和rotation 轨迹
    """

    # --- 1. 解析输入 ---
    if isinstance(small_box_data_input, dict):
        small_box_data_list = [small_box_data_input]
    else:
        small_box_data_list = small_box_data_input

        # 解析大箱子起点
    big_start_pos = big_start_pose[0]  # (3,)
    big_start_rot_mat = big_start_pose[1]  # (3, 3)

    # 获取时间步数 N (假设所有轨迹长度一致，取第一条的长度)
    # 如果轨迹长度不一致，ProMP 内部其实是按归一化时间 t=[0,1] 处理的，这里主要为了生成 N
    ref_N = len(small_box_data_list[0]['pos'])

    # ==========================================
    # Step 1: 数据对齐 (Pre-Alignment) -> 提取纯形状
    # ==========================================
    canonical_deviations = []

    for traj_data in small_box_data_list:
        dev_canonical = TrajectoryUtils.transaction_trajectory_data_alignment(traj_data)
        canonical_deviations.append(dev_canonical)

    # ==========================================
    # Step 2: ProMP 训练 (在标准坐标系下)
    # ==========================================
    promp = SimpleProMP(n_basis=n_basis, n_dims=3)
    promp.train(canonical_deviations)

    # ==========================================
    # Step 3: ProMP 生成 (Canonical Frame)
    # ==========================================
    # 生成标准的、沿着 X 轴的偏差
    gen_dev_canonical = promp.generate_trajectory(
        duration=1.0, num_steps=ref_N,
        start_pos=np.array([0, 0, 0]), end_pos=np.array([0, 0, 0])
    )

    # ==========================================
    # Step 4: 应用到新任务 (Post-Alignment)
    # ==========================================
    new_trans_traj = TrajectoryUtils.apply_deviation_2_new_line(big_start_pos, big_end_pos, gen_dev_canonical, ref_N)

    # ==========================================
    # Step B: 旋转部分:这部分也比较复杂
    # ==========================================

    all_continuous_rotvecs = []
    for traj_data in small_box_data_list:
        continuous_rot_vecs = TrajectoryUtils.rotation_trajectory_data_alignment(traj_data, single_arm_base_pose)

        all_continuous_rotvecs.append(continuous_rot_vecs)

    promp_rot = SimpleProMP(n_basis=n_basis, n_dims=3)
    promp_rot.train(all_continuous_rotvecs)

    gen_rotvec_details = promp_rot.generate_trajectory(
        duration=1.0, num_steps=ref_N
        # 旋转不需要像位置一样强制归零，它通常代表姿态的变化趋势
    )

    rot_details_promp = R.from_rotvec(gen_rotvec_details)

    r_final_traj = TrajectoryUtils.apply_rot_details_2_new_base_rot(rot_details_promp, new_trans_traj,
                                                                    dual_arm_robot_base_pos, big_start_rot_mat)

    return {'pos': new_trans_traj, 'rot': r_final_traj.as_matrix()}


# ==========================================
# 控制执行函数 (Robust版)
# ==========================================

def execute_trajectory_general(trajectories,
                               configuration, tasks, limits, solver,
                               model, data, viewer, rate,
                               arm_infos, grap=False, gripper_vals=None, record_body_id=None):
    """
    通用执行函数。
    """
    # 1. 格式归一化
    # 如果 trajectories[0] 不是 list (说明传入的是扁平的 [SE3, SE3...])，则包裹它
    if len(trajectories) > 0 and not isinstance(trajectories[0], list):
        trajectories = [trajectories]

    arm_infos = Format.ensure_list(arm_infos)

    # 确保 gripper_vals 长度足够
    num_arms = len(arm_infos)
    if gripper_vals is None:
        gripper_vals = [0] * num_arms
    else:
        gripper_vals = Format.ensure_list(gripper_vals)
        if len(gripper_vals) < num_arms:
            # 如果只传了一个值但有两个手臂，复制该值
            gripper_vals = [gripper_vals[0]] * num_arms

    physics_steps = int(np.ceil(rate.dt / model.opt.timestep))

    # 轨迹记录list初始化
    rec_pos = []
    rec_rot = []

    # 2. 执行循环
    # zip(*traj) 相当于矩阵转置，把 [[arm1_t1, arm1_t2], [arm2_t1, arm2_t2]]
    # 变成 [(arm1_t1, arm2_t1), (arm1_t2, arm2_t2)]
    for targets_at_step in zip(*trajectories):
        if not viewer.is_running(): break

        # 设置目标
        for i, target in enumerate(targets_at_step):
            # 防御：确保 tasks 数量足够
            if i < len(tasks):
                tasks[i].set_target(target)

        # IK
        configuration.update(data.qpos)
        vel = mink.solve_ik(configuration, tasks, rate.dt, solver, 1e-3, limits=limits)
        configuration.integrate_inplace(vel, rate.dt)

        # Control
        for i, info in enumerate(arm_infos):
            c_idx, q_idx, dof = info['ctrl_idx'], info['qpos_idx'], info['range']
            data.ctrl[c_idx: c_idx + dof] = configuration.q[q_idx: q_idx + dof]

            if grap and info['gripper_ctrl_idx'] is not None:
                data.ctrl[info['gripper_ctrl_idx']] = gripper_vals[i]

        # Physics
        for _ in range(physics_steps):
            # 简单的重力补偿
            for info in arm_infos:
                c_idx, dof = info['ctrl_idx'], info['range']
                data.qfrc_applied[c_idx: c_idx + dof] = data.qfrc_bias[c_idx: c_idx + dof]
            mujoco.mj_step(model, data)

        if record_body_id is not None:
            # 记录位置 (3,)
            rec_pos.append(data.xpos[record_body_id].copy())
            # 记录旋转 (3, 3) - 注意 xmat 是扁平的，需要 reshape
            rec_rot.append(data.xmat[record_body_id].copy().reshape(3, 3))

        viewer.sync()
        rate.sleep()

    if record_body_id is not None:
        return {
            "pos": np.array(rec_pos),  # Shape: (N, 3)
            "rot": np.array(rec_rot)  # Shape: (N, 3, 3)
        }
    return None


def hold_current_position_and_open_or_close_gripper(hold_target_se3_list, physics_steps, hold_time, gripper_vals,
                                                    configuration, tasks, limits, solver,
                                                    model, data, viewer, rate, arm_infos, record_body_id=None):
    """
    保持位置函数。
    注意：hold_target_se3_list 必须是 [SE3_arm1, SE3_arm2] 这样的列表，而不是一整条轨迹！
    """
    hold_steps = int(hold_time / rate.dt)

    # 1. 格式归一化
    hold_target_se3_list = Format.ensure_list(hold_target_se3_list)
    arm_infos = Format.ensure_list(arm_infos)
    gripper_vals = Format.ensure_list(gripper_vals)
    if len(gripper_vals) < len(arm_infos):
        gripper_vals = [gripper_vals[0]] * len(arm_infos)

    # 2. 设置目标 (只设置一次)
    for i, target in enumerate(hold_target_se3_list):
        # 关键修正：确保只给 FrameTask 设置 SE3 目标，不要设置给 PostureTask
        # 假设 tasks 列表的前 N 个对应 N 个手臂
        if i < len(tasks) and not isinstance(tasks[i], mink.PostureTask):
            tasks[i].set_target(target)

    # 获取物体轨迹
    rec_pos = []
    rec_rot = []

    # 3. 保持循环
    for _ in range(hold_steps):
        if not viewer.is_running(): break

        configuration.update(data.qpos)
        vel = mink.solve_ik(configuration, tasks, rate.dt, solver, 1e-3, limits=limits)
        configuration.integrate_inplace(vel, rate.dt)

        for i, info in enumerate(arm_infos):
            c_idx, q_idx, dof = info['ctrl_idx'], info['qpos_idx'], info['range']
            g_idx = info['gripper_ctrl_idx']

            data.ctrl[c_idx: c_idx + dof] = configuration.q[q_idx: q_idx + dof]
            if g_idx is not None:
                data.ctrl[g_idx] = gripper_vals[i]

        for _ in range(physics_steps):
            for info in arm_infos:
                c_idx, dof = info['ctrl_idx'], info['range']
                data.qfrc_applied[c_idx: c_idx + dof] = data.qfrc_bias[c_idx: c_idx + dof]
            mujoco.mj_step(model, data)

        if record_body_id is not None:
            rec_pos.append(data.xpos[record_body_id].copy())
            rec_rot.append(data.xmat[record_body_id].copy().reshape(3, 3))

        viewer.sync()
        rate.sleep()

    if record_body_id is not None:
        return {
            "pos": np.array(rec_pos),
            "rot": np.array(rec_rot)
        }
    return None


def wait_for_convergence(singleArm, target_pos_xyz, target_se3_pose, gripper_vals,
                         control_params, threshold=5e-4, record_body_id=None):
    (model, data, physics_steps, hold_time, config, tasks, limits,
     solver, viewer, rate, arm_infos) = control_params

    # 确保 target_se3_pose 是列表，用于 hold 函数
    target_se3_list = Format.ensure_list(target_se3_pose)

    current_position, _ = singleArm.get_xpos_and_rot(data)
    error = np.linalg.norm(current_position - target_pos_xyz)
    print(f"开始收敛... 初始误差: {error:.4f} m")

    # 用于收集等待收敛时候的数据
    convergence_segments = []

    while error > threshold:
        if not viewer.is_running(): break

        # 这里的 hold_time 可以设短一点，比如 1 个 dt，或者 0.1s
        segment = hold_current_position_and_open_or_close_gripper(
            target_se3_list, physics_steps, 0.05, gripper_vals,  # 每次 hold 0.05秒
            config, tasks, limits, solver,
            model, data, viewer, rate, arm_infos, record_body_id
        )
        if segment is not None:
            convergence_segments.append(segment)

        current_position, _ = singleArm.get_xpos_and_rot(data)
        error = np.linalg.norm(current_position - target_pos_xyz)
        # print(f"误差: {error:.4f} m")

    print(f"位置已收敛。")

    # 如果有记录数据，合并后返回
    if record_body_id is not None and convergence_segments:
        return TrajectoryUtils.merge_trajectory_data(convergence_segments)

    return None


def execute_object_centric_trajectory(object_trajectory,
                                      T_obj_to_left, T_obj_to_right,
                                      configuration, tasks, limits, solver,
                                      model, data, viewer, rate,
                                      arm_infos, gripper_vals):
    """
    Method 3 核心实现：
    输入物体的轨迹，利用相对变换算出左右手的目标，统一求解 IK。
    """

    gripper_vals = Format.ensure_list(gripper_vals)
    if len(gripper_vals) < len(arm_infos):
        gripper_vals = [gripper_vals[0]] * len(arm_infos)

    physics_steps = int(np.ceil(rate.dt / model.opt.timestep))

    # 这里的 tasks[0] 对应左臂, tasks[1] 对应右臂 (根据你main里的定义顺序)
    task_left = tasks[0]
    task_right = tasks[1]

    for T_world_obj in object_trajectory:
        if not viewer.is_running(): break

        # --- 核心逻辑 Start ---
        # 根据当前物体目标的位姿，反算左右手应该在哪
        # T_world_hand = T_world_obj * T_obj_hand
        target_left = T_world_obj.multiply(T_obj_to_left)
        target_right = T_world_obj.multiply(T_obj_to_right)

        # 同时设置两个目标
        task_left.set_target(target_left)
        task_right.set_target(target_right)
        # --- 核心逻辑 End ---

        # 统一求解 IK (Coupled IK)
        configuration.update(data.qpos)
        vel = mink.solve_ik(configuration, tasks, rate.dt, solver, 1e-3, limits=limits)
        configuration.integrate_inplace(vel, rate.dt)

        # 控制写入
        for i, info in enumerate(arm_infos):
            c_idx, q_idx, dof = info['ctrl_idx'], info['qpos_idx'], info['range']
            data.ctrl[c_idx: c_idx + dof] = configuration.q[q_idx: q_idx + dof]

            # 保持夹爪力
            if info['gripper_ctrl_idx'] is not None:
                data.ctrl[info['gripper_ctrl_idx']] = gripper_vals[i]

        # 物理步进
        for _ in range(physics_steps):
            for info in arm_infos:
                c_idx, dof = info['ctrl_idx'], info['range']
                data.qfrc_applied[c_idx: c_idx + dof] = data.qfrc_bias[c_idx: c_idx + dof]
            mujoco.mj_step(model, data)

        viewer.sync()
        rate.sleep()


def wait_for_object_convergence(target_obj_se3, object: SceneObject,
                                T_obj_to_left, T_obj_to_right,
                                configuration, tasks, limits, solver,
                                model, data, viewer, rate,
                                arm_infos, gripper_vals,
                                max_wait_time=3.0,  # 最大等待时间 (秒)
                                pos_threshold=0.005,  # 位置误差阈值 (5mm)
                                rot_threshold_deg=2.0  # 旋转误差阈值 (2度)
                                ):
    """
    Method 3 的专用收敛函数。
    持续维持闭链控制，直到物体实际位姿与 target_obj_se3 的误差小于阈值。
    """
    print(f"开始等待物体 [{object.body_name}] 收敛...")

    # 确保 gripper_vals 格式正确 (复用之前的修复逻辑)
    gripper_vals = Format.ensure_list(gripper_vals)
    num_grippers = sum(1 for info in Format.ensure_list(arm_infos) if info['gripper_ctrl_idx'] is not None)
    if len(gripper_vals) == 1 and num_grippers > 1:
        gripper_vals = [gripper_vals[0]] * num_grippers

    physics_steps = int(np.ceil(rate.dt / model.opt.timestep))

    # 设置固定的目标 (在等待期间目标不变)
    task_left = tasks[0]
    task_right = tasks[1]

    target_left = target_obj_se3.multiply(T_obj_to_left)
    target_right = target_obj_se3.multiply(T_obj_to_right)

    task_left.set_target(target_left)
    task_right.set_target(target_right)

    start_time = time.time()
    is_converged = False

    final_pos_err = 0.0
    final_rot_err = 0.0

    while (time.time() - start_time) < max_wait_time:
        if not viewer.is_running(): break

        # 1. 获取当前物体位姿
        curr_obj_se3 = object.get_se3()

        # 2. 计算误差
        final_pos_err, final_rot_err = calculate_pose_error(curr_obj_se3, target_obj_se3)

        # 3. 检查是否收敛
        if final_pos_err < pos_threshold and final_rot_err < rot_threshold_deg:
            print(f"收敛成功! 用时: {time.time() - start_time:.2f}s")
            is_converged = True
            break

        # 4. 继续运行 IK 和 控制 (保持姿态抵抗重力)
        configuration.update(data.qpos)
        vel = mink.solve_ik(configuration, tasks, rate.dt, solver, 1e-3, limits=limits)
        configuration.integrate_inplace(vel, rate.dt)

        gripper_counter = 0
        for i, info in enumerate(arm_infos):
            c_idx, q_idx, dof = info['ctrl_idx'], info['qpos_idx'], info['range']
            data.ctrl[c_idx: c_idx + dof] = configuration.q[q_idx: q_idx + dof]

            if info['gripper_ctrl_idx'] is not None:
                if gripper_counter < len(gripper_vals):
                    data.ctrl[info['gripper_ctrl_idx']] = gripper_vals[gripper_counter]
                    gripper_counter += 1

        for _ in range(physics_steps):
            for info in arm_infos:
                c_idx, dof = info['ctrl_idx'], info['range']
                data.qfrc_applied[c_idx: c_idx + dof] = data.qfrc_bias[c_idx: c_idx + dof]
            mujoco.mj_step(model, data)

        viewer.sync()
        rate.sleep()

    # 打印最终报告
    if not is_converged:
        print(f"警告: 超时未完全收敛 ({max_wait_time}s)。")

    print(f"最终误差 -> 位置: {final_pos_err * 1000:.2f} mm, 角度: {final_rot_err:.2f} deg")

    return is_converged, final_pos_err, final_rot_err


# ==========================================
# Main
# ==========================================

def init_mink(configuration, max_velocity, *joint_names, **task_frame):
    max_velocities = {name: max_velocity for name in joint_names}
    limits = [
        mink.ConfigurationLimit(model=configuration.model),
        mink.VelocityLimit(model, max_velocities)
    ]

    tasks = []
    for frame_name, frame_type in task_frame.items():
        task = mink.FrameTask(
            frame_name=frame_name,
            frame_type=frame_type,
            position_cost=10.0,
            orientation_cost=1.0,
            lm_damping=1.0
        )
        tasks.append(task)

    posture_task = mink.PostureTask(model, cost=1e-3)
    posture_task.set_target(configuration.q)
    tasks.append(posture_task)
    return tasks, limits


def compare_distant_error(current_pos, target_pos):
    return np.linalg.norm(current_pos - target_pos)


def calculate_pose_error(current_se3, target_se3):
    """
    计算两个 SE3 之间的误差。
    Returns:
        pos_error (float): 欧几里得距离 (m)
        rot_error_deg (float): 旋转角度差 (degree)
    """
    # 1. 位置误差: || p_curr - p_target ||
    pos_diff = current_se3.translation() - target_se3.translation()
    pos_error = np.linalg.norm(pos_diff)

    # 2. 旋转误差: 计算 R_diff = R_target * inv(R_curr) 的旋转角度
    #    利用 Scipy 的 magnitude() 方法最简单
    r_curr = R.from_matrix(current_se3.rotation().as_matrix())
    r_target = R.from_matrix(target_se3.rotation().as_matrix())

    # 计算相对旋转
    r_diff = r_target * r_curr.inv()
    rot_error_rad = r_diff.magnitude()  # 返回弧度 [0, pi]
    rot_error_deg = np.rad2deg(rot_error_rad)

    return pos_error, rot_error_deg


if __name__ == "__main__":

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    configuration = mink.Configuration(model)

    print("正在进行物理预热，让箱子自然落位...")

    # 1. 保存机器人的初始姿态 (因为预热时机器人可能会因为重力下垂)
    #    假设你后面定义的 single_arm_init_q 是你想要的初始状态
    #    为了安全，我们先在这里手动定义一下，或者你把后面那行代码提上来
    temp_init_q = np.array([-0.329, 0.429, -0.261, -0.805, 0.231, 1.19, 0.435])

    # 2. 获取 single arm 的 qpos 地址 (为了复位用)
    #    (你原本的代码是在后面获取的，我们需要把这几行简单的获取逻辑提上来)
    sa_joint1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint1")
    sa_qpos_adr = model.jnt_qposadr[sa_joint1_id]

    # 3. 后台跑 500 步物理仿真 (不渲染，速度极快，几毫秒就完成了)
    #    这会让箱子从 Z=0.121 掉落并稳定在 Z=0.061
    for _ in range(500):
        mujoco.mj_step(model, data)

    # 4. 关键步骤：重置机器人状态，但保留箱子的状态
    #    因为刚才跑了 mj_step，没加控制的机器人可能已经瘫软了
    #    我们要把它“拉回”到初始位置，但箱子已经掉落在桌子上了，不要动箱子
    data.qvel[:] = 0  # 清空所有速度（防止机器人复位后带有动量）
    data.qpos[sa_qpos_adr:sa_qpos_adr + 7] = temp_init_q  # 强行把机器人按回原位

    # 5. 刷新运动学数据 (Forward Kinematics)
    #    这一步非常重要！它会更新 data.xpos, data.site_xpos 等所有坐标
    mujoco.mj_forward(model, data)

    print("预热完成。箱子已在地面，机器人已复位。")

    sim_dt = model.opt.timestep  # 通常是 0.002s
    control_freq = 60.0
    control_dt = 1.0 / control_freq
    physics_steps_per_control_step = int(np.ceil(control_dt / sim_dt))
    solver = "daqp"
    mujoco.mj_forward(model, data)

    single_arm_attachment_site_name = "panda_attachment_site"
    # 双臂的左、右 site 点名字
    ur5_left_arm_attachment_site_name = 'ur_grip_site_left'
    ur5_right_arm_attachment_site_name = 'ur_grip_site_right'

    single_arm_joints_name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7']
    ur5_left_arm_joint_names = ['joint0_ur5left', 'joint1_ur5left', 'joint2_ur5left', 'joint3_ur5left',
                                'joint4_ur5left', 'joint5_ur5left']
    ur5_right_arm_joint_names = ['joint0_ur5right', 'joint1_ur5right', 'joint2_ur5right', 'joint3_ur5right',
                                 'joint4_ur5right', 'joint5_ur5right']

    base_joint_name = ['ur_stand_joint']

    single_arm_first_actuator_name = 'actuator1'
    ur5_left_arm_first_actuator_name = 'joint0_motor_ur5left'
    ur5_right_arm_first_actuator_name = 'joint0_motor_ur5right'
    base_actuator_name = 'ur_stand_joint_motor'

    single_arm_gripper_actuator_name = 'actuator8'
    ur5_left_arm_gripper_actuator_name = 'gripper_ur5left'
    ur5_right_arm_gripper_actuator_name = 'gripper_ur5right'

    single_arm_init_q = np.array([-0.329, 0.429, -0.261, -0.805, 0.231, 1.19, 0.435])

    singleArm = RobotArm('single', single_arm_joints_name, single_arm_attachment_site_name,
                         single_arm_first_actuator_name, single_arm_gripper_actuator_name, single_arm_init_q)
    singleArm.initialize(model)
    singleArm.reset_to_home(data)

    ur5LeftArm = RobotArm('ur5_left', ur5_left_arm_joint_names, ur5_left_arm_attachment_site_name,
                          ur5_left_arm_first_actuator_name, ur5_left_arm_gripper_actuator_name)

    ur5RightArm = RobotArm('ur5_right', ur5_right_arm_joint_names, ur5_right_arm_attachment_site_name,
                           ur5_right_arm_first_actuator_name, ur5_right_arm_gripper_actuator_name)

    ur5BaseBody = RobotArm('base', base_joint_name, None, base_actuator_name)

    ur5LeftArm.initialize(model)
    ur5RightArm.initialize(model)
    ur5BaseBody.initialize(model)

    single_arm_final_rotation = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ])
    ur5_left_arm_final_rotation = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [-1, 0, 0]
    ])

    ur5_right_arm_final_rotation = np.array([
        [0, 0, -1],
        [0, 1, 0],
        [1, 0, 0]
    ])

    # 两个物体在世界的坐标系位置
    small_box_body_name = "small_box"
    big_box_body_name = "big_box"
    small_box_geom_name = "small_box_geom"
    big_box_geom_name = "big_box_geom"

    SmallBox = SceneObject(small_box_body_name, model, data, small_box_geom_name)
    BigBox = SceneObject(big_box_body_name, model, data, big_box_geom_name)

    small_box_point = SmallBox.get_feature_points()

    big_box_point = BigBox.get_feature_points()

    # 没动box 之前先算一个affine 转化矩阵出来，后面获取了抓取点后好计算
    A_aff, t_aff = SmallBox.compute_affine_transform_to(BigBox)

    small_box_xpos, _ = SmallBox.get_xpos_and_rot()
    big_box_xpos, _ = BigBox.get_xpos_and_rot()

    # 设置单机械臂移动到小box上方的位置坐标
    single_target_position_1 = small_box_xpos + [0, 0, 0.1]
    print('the first target position is ' + str(single_target_position_1))
    target_positions = [single_target_position_1]

    mujoco.mj_forward(model, data)

    single_arm_start_xpos, single_arm_start_rotation = singleArm.get_xpos_and_rot(data)

    linear_speed = 0.5
    slow_speed = 0.2
    print('the start pose is ' + str(single_arm_start_xpos))
    print('the end pose is ' + str(single_target_position_1))

    single_arm_se3_trajectories = generate_multi_cartesian_trajectories(start_pos_list=single_arm_start_xpos,
                                                                        end_pos_list=target_positions[0],
                                                                        start_rot_list=single_arm_start_rotation,
                                                                        end_rot_list=single_arm_final_rotation,
                                                                        linear_speed=linear_speed,
                                                                        control_dt=1 / 60)

    single_arm_infos = [singleArm.get_control_info()]
    ur5_arm_infos = [ur5BaseBody.get_control_info(), ur5LeftArm.get_control_info(), ur5RightArm.get_control_info()]
    single_arm_task_info = {single_arm_attachment_site_name: 'site'}
    single_arm_tasks, single_arm_limits = init_mink(configuration, np.pi, *single_arm_joints_name,
                                                    **single_arm_task_info)

    with mujoco.viewer.launch_passive(
            model=model,
            data=data,
            show_left_ui=False,
            show_right_ui=False,
    ) as viewer:
        viewer.cam.lookat[:] = [1.0, 0.0, 0.5]
        # mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        viewer.cam.azimuth = -90
        # 3. 调整俯视角度
        viewer.cam.elevation = -15
        # 4. 调整距离
        viewer.cam.distance = 5.0
        rate = RateLimiter(frequency=control_freq, warn=False)
        viewer.sync()

        single_arm_hold_params = (
            model, data, physics_steps_per_control_step, 2, configuration, single_arm_tasks, single_arm_limits,
            solver,
            viewer, rate, single_arm_infos)
        # 执行轨迹
        execute_trajectory_general(single_arm_se3_trajectories, configuration, single_arm_tasks, single_arm_limits,
                                   solver,
                                   model, data, viewer,
                                   rate, single_arm_infos)

        # 等待一下 机械臂到规定的位置
        final_target_se3 = single_arm_se3_trajectories[0][-1]
        wait_for_convergence(singleArm, single_target_position_1, final_target_se3, 0, single_arm_hold_params)

        print('开始旋转夹爪')
        single_arm_rotation_spac = [singleArm.set_rotation_info(-90)]
        single_arm_rotation_trajectories = plan_multi_gripper_rotation(data, single_arm_rotation_spac, dt=rate.dt)
        execute_trajectory_general(single_arm_rotation_trajectories, configuration, single_arm_tasks, single_arm_limits,
                                   solver,
                                   model, data,
                                   viewer, rate, single_arm_infos)

        # 开始打开夹爪
        print('打开夹爪')
        curr_pos, _ = singleArm.get_xpos_and_rot(data)
        hold_current_position_and_open_or_close_gripper(single_arm_rotation_trajectories[0][-1],
                                                        physics_steps_per_control_step, 2,
                                                        0.04,
                                                        configuration, single_arm_tasks, single_arm_limits, solver,
                                                        model, data, viewer, rate, single_arm_infos)

        # 下降一段距离
        pre_pick_pos, pre_pick_rot_mat = singleArm.get_xpos_and_rot(data)
        pick_pos = pre_pick_pos - [0, 0, 0.09]
        pick_trajectory = generate_multi_cartesian_trajectories(start_pos_list=pre_pick_pos,
                                                                end_pos_list=pick_pos,
                                                                start_rot_list=pre_pick_rot_mat,
                                                                end_rot_list=pre_pick_rot_mat,
                                                                linear_speed=linear_speed,
                                                                control_dt=rate.dt)

        execute_trajectory_general(pick_trajectory, configuration, single_arm_tasks, single_arm_limits, solver, model,
                                   data, viewer, rate, single_arm_infos)

        # 关闭夹爪
        final_pick_trajectory = pick_trajectory[0][-1]
        hold_current_position_and_open_or_close_gripper(final_pick_trajectory, physics_steps_per_control_step, 1, 0.04,
                                                        configuration, single_arm_tasks, single_arm_limits, solver,
                                                        model, data, viewer, rate, single_arm_infos)

        hold_current_position_and_open_or_close_gripper(final_pick_trajectory, physics_steps_per_control_step, 1, 0,
                                                        configuration, single_arm_tasks, single_arm_limits, solver,
                                                        model, data, viewer, rate, single_arm_infos)

        # 关闭夹爪之后，计算夹爪的position，用于transfer到大物体上
        # 获取 Geom 的世界坐标 (xpos)
        left_pad_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_finger_pad")
        right_pad_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_finger_pad")
        # data.geom_xpos 存储的是每个 geom 的全局位置
        left_pad_pos = data.geom_xpos[left_pad_id]
        right_pad_pos = data.geom_xpos[right_pad_id]
        print('the pick pose is ' + str(left_pad_pos) + '  ' + str(right_pad_pos))
        small_box_pickup_points = np.array([left_pad_pos, right_pad_pos])

        # 夹起物体上升一段距离
        print('开始上升')
        curr_pos, curr_rot_mat = singleArm.get_xpos_and_rot(data)

        up_pos = curr_pos + [0, 0, 0.1]
        up_rot_mat = curr_rot_mat

        pre_place_pos = [2.5, 0, 0.5]

        pre_place_rotation = apply_local_rotation(curr_rot_mat, 'z', 90)

        place_pos = [2.5, 0, 0.42]

        full_action_segments = []

        up_trajectory = generate_multi_cartesian_trajectories(start_pos_list=curr_pos,
                                                              end_pos_list=up_pos,
                                                              start_rot_list=curr_rot_mat,
                                                              end_rot_list=up_rot_mat,
                                                              linear_speed=slow_speed,
                                                              control_dt=rate.dt)

        arch_trajectory = generate_multi_cartesian_trajectories(start_pos_list=up_pos,
                                                                end_pos_list=pre_place_pos,
                                                                start_rot_list=up_rot_mat,
                                                                end_rot_list=pre_place_rotation,
                                                                linear_speed=slow_speed,
                                                                control_dt=rate.dt,
                                                                peak_height=0.2)

        place_trajectory = generate_multi_cartesian_trajectories(start_pos_list=pre_place_pos,
                                                                 end_pos_list=place_pos,
                                                                 start_rot_list=pre_place_rotation,
                                                                 end_rot_list=pre_place_rotation,
                                                                 linear_speed=slow_speed,
                                                                 control_dt=rate.dt,
                                                                 peak_height=0)

        all_trajectory = [up_trajectory[0] + arch_trajectory[0]]

        pick_up_small_box_trajectory = execute_trajectory_general(all_trajectory, configuration, single_arm_tasks,
                                                                  single_arm_limits, solver, model,
                                                                  data, viewer, rate, single_arm_infos, True, [0],
                                                                  SmallBox.body_id)

        full_action_segments.append(pick_up_small_box_trajectory)

        final_up_position = all_trajectory[0][-1]
        hold_params = (
            model, data, physics_steps_per_control_step, 2, configuration, single_arm_tasks, single_arm_limits,
            solver,
            viewer, rate, single_arm_infos)
        wait_segments = wait_for_convergence(singleArm, pre_place_pos, final_up_position, 0, hold_params, 5e-4,
                                             SmallBox.body_id)

        full_action_segments.append(wait_segments)

        small_box_down_trajectory_1 = execute_trajectory_general(place_trajectory, configuration, single_arm_tasks,
                                                                 single_arm_limits, solver, model,
                                                                 data, viewer, rate, single_arm_infos, True, [0],
                                                                 SmallBox.body_id)

        final_place_position = place_trajectory[0][-1]

        small_box_down_trajectory_2 = hold_current_position_and_open_or_close_gripper(final_place_position,
                                                                                      physics_steps_per_control_step, 2,
                                                                                      0,
                                                                                      configuration, single_arm_tasks,
                                                                                      single_arm_limits, solver,
                                                                                      model, data, viewer, rate,
                                                                                      single_arm_infos,
                                                                                      SmallBox.body_id)

        hold_current_position_and_open_or_close_gripper(final_place_position, physics_steps_per_control_step, 2, 0.04,
                                                        configuration, single_arm_tasks, single_arm_limits, solver,
                                                        model, data, viewer, rate, single_arm_infos)

        full_action_segments.append(small_box_down_trajectory_1)
        full_action_segments.append(small_box_down_trajectory_2)

        # 生成small box 的轨迹,保存后可视化
        final_box_trajectory = TrajectoryUtils.merge_trajectory_data(full_action_segments)

        if final_box_trajectory is not None:
            print(f"完整轨迹获取成功，共 {len(final_box_trajectory['pos'])} 帧")

            np.savez("data/small_box_trajectory_data.npz",
                     pos=final_box_trajectory['pos'],
                     rot=final_box_trajectory['rot'])
            print("数据已保存到 small_box_trajectory_data.npz，请运行 plot_script.py 查看结果。")
        else:
            print("没有数据。")

        big_box_start_pos, big_box_start_rot = BigBox.get_xpos_and_rot()
        big_box_start_pose = (big_box_start_pos, big_box_start_rot)
        big_box_end_pos = np.array([-0.5, 0.4, 0.3])
        # big_box_trajectory = generate_big_box_trajectory(final_box_trajectory, big_box_start_pose, big_box_end_pos)
        big_box_trajectory = generate_big_box_trajectory_promp(final_box_trajectory, big_box_start_pose,
                                                               big_box_end_pos, rot_scale=1)
        if big_box_trajectory is not None:
            print(f"大物体轨迹生成成功，共 {len(big_box_trajectory['pos'])} 帧")

            gen_start = big_box_trajectory['pos'][0]
            gen_end = big_box_trajectory['pos'][-1]

            print(f"目标起点: {big_box_start_pose[0]}")
            print(f"生成起点: {gen_start}")
            print(f"起点误差: {np.linalg.norm(gen_start - big_box_start_pose[0])}")

            print(f"目标终点: {big_box_end_pos}")
            print(f"生成终点: {gen_end}")
            print(f"终点误差: {np.linalg.norm(gen_end - big_box_end_pos)}")

            np.savez("data/big_box_trajectory_data.npz",
                     pos=big_box_trajectory['pos'],
                     rot=big_box_trajectory['rot'])
            print("数据已保存到 big_box_trajectory_data.npz，请运行 plot_script.py 查看结果。")
        else:
            print("没有数据。")

        # 开始双臂机器的搬运工作
        print('开始映射small box 的抓取点到big box')
        big_box_pickup_points = small_box_pickup_points @ A_aff.T + t_aff
        print('the big box pick up points ' + str(big_box_pickup_points))

        pre_big_box_pickup_points = [big_box_pickup_points[0] + [0.1, 0, 0], big_box_pickup_points[1] - [0.1, 0, 0]]
        pre_big_box_pickup_points.reverse()

        print('开始双臂抓取任务')
        ur5_all_joints_names = base_joint_name + ur5_left_arm_joint_names + ur5_right_arm_joint_names
        ur5_tasks_info = {ur5_left_arm_attachment_site_name: 'site', ur5_right_arm_attachment_site_name: 'site'}
        ur5_tasks, ur5_limits = init_mink(configuration, np.pi, *ur5_all_joints_names, **ur5_tasks_info)

        ur5_left_arm_init_xpos, ur5_left_arm_init_rot = ur5LeftArm.get_xpos_and_rot(data)

        ur5_right_arm_init_xpos, ur5_right_arm_init_rot = ur5RightArm.get_xpos_and_rot(data)
        ur5_all_arm_init_rotations = [ur5_left_arm_init_rot, ur5_right_arm_init_rot]
        ur5_all_arm_final_rotations = [ur5_left_arm_final_rotation, ur5_right_arm_final_rotation]

        # 计算两个机械臂的移动轨迹
        ur5_all_arms_trajectories = generate_multi_cartesian_trajectories(
            start_pos_list=[ur5_left_arm_init_xpos, ur5_right_arm_init_xpos],
            end_pos_list=pre_big_box_pickup_points,
            start_rot_list=ur5_all_arm_init_rotations,
            end_rot_list=ur5_all_arm_final_rotations,
            linear_speed=linear_speed,
            control_dt=rate.dt)

        execute_trajectory_general(ur5_all_arms_trajectories, configuration, ur5_tasks, ur5_limits, solver,
                                   model, data, viewer, rate, ur5_arm_infos)

        ur5_pre_pick_hold_pos = [ur5_all_arms_trajectories[0][-1], ur5_all_arms_trajectories[1][-1]]
        hold_current_position_and_open_or_close_gripper(ur5_pre_pick_hold_pos, physics_steps_per_control_step, 2,
                                                        [-1, -1],
                                                        configuration, ur5_tasks, ur5_limits, solver,
                                                        model, data, viewer, rate,
                                                        ur5_arm_infos)

        # 旋转夹爪角度
        ur5_all_arms_rotation_spac = [ur5LeftArm.set_rotation_info(-90), ur5RightArm.set_rotation_info(90)]

        ur5_all_rotation_trajectories = plan_multi_gripper_rotation(data, ur5_all_arms_rotation_spac, dt=rate.dt)
        execute_trajectory_general(ur5_all_rotation_trajectories, configuration, ur5_tasks, ur5_limits, solver,
                                   model, data,
                                   viewer, rate, ur5_arm_infos)

        ur5_pre_pick_rotation_pos = [ur5_all_rotation_trajectories[0][-1], ur5_all_rotation_trajectories[1][-1]]

        hold_current_position_and_open_or_close_gripper(ur5_pre_pick_rotation_pos, physics_steps_per_control_step, 2,
                                                        [-1, -1],
                                                        configuration, ur5_tasks, ur5_limits, solver,
                                                        model, data, viewer, rate,
                                                        ur5_arm_infos)
        # 进入夹取position
        ur5_pick_up_points = [big_box_pickup_points[1], big_box_pickup_points[0]]
        ur5_all_arms_current_pos, ur5_all_arms_current_rot = TrajectoryUtils.get_current_pos_and_rot(data, ur5LeftArm,
                                                                                                     ur5RightArm)

        ur5_all_arms_pick_trajectories = generate_multi_cartesian_trajectories(
            start_pos_list=ur5_all_arms_current_pos,
            end_pos_list=ur5_pick_up_points,
            start_rot_list=ur5_all_arms_current_rot,
            end_rot_list=ur5_all_arms_current_rot,
            linear_speed=linear_speed,
            control_dt=rate.dt)

        execute_trajectory_general(ur5_all_arms_pick_trajectories, configuration, ur5_tasks, ur5_limits, solver,
                                   model, data, viewer, rate, ur5_arm_infos)

        ur5_pick_hold_pos = [ur5_all_arms_pick_trajectories[0][-1], ur5_all_arms_pick_trajectories[1][-1]]

        hold_current_position_and_open_or_close_gripper(ur5_pick_hold_pos, physics_steps_per_control_step, 2,
                                                        [-1, -1],
                                                        configuration, ur5_tasks, ur5_limits, solver,
                                                        model, data, viewer, rate,
                                                        ur5_arm_infos)

        # 打印一下关闭夹爪前的夹爪位置
        ur5_all_arms_current_pos, _ = TrajectoryUtils.get_current_pos_and_rot(data, ur5LeftArm, ur5RightArm)
        print('current ur5 position is' + str(ur5_all_arms_current_pos))

        # 关闭夹爪
        hold_current_position_and_open_or_close_gripper(ur5_pick_hold_pos, physics_steps_per_control_step, 1,
                                                        [1, 1],
                                                        configuration, ur5_tasks, ur5_limits, solver,
                                                        model, data, viewer, rate,
                                                        ur5_arm_infos)

        # 移动物体
        # 关闭之后，需要获取两个夹爪当前和物体质心的转换矩阵
        curr_box_se3 = BigBox.get_se3()
        curr_left_se3 = ur5LeftArm.get_se3(data)
        curr_right_se3 = ur5RightArm.get_se3(data)

        # 计算物体质心到机械臂 site点的 transfer矩阵
        T_obj_to_left = curr_box_se3.inverse().multiply(curr_left_se3)
        T_obj_to_right = curr_box_se3.inverse().multiply(curr_right_se3)

        # 计算两个夹爪之间的相对 transfer
        T_rel_target = T_obj_to_left.inverse().multiply(T_obj_to_right)

        task_relative = mink.RelativeFrameTask(
            frame_name=ur5RightArm.attachment_site_name,  # 被约束的 Frame (右手)
            frame_type="site",  # 类型是 site
            root_name=ur5LeftArm.attachment_site_name,  # 参考系 Root (左手)
            root_type="site",  # 类型是 site
            position_cost=50.0,  # 强位置约束
            orientation_cost=5.0,  # 强旋转约束
            lm_damping=1.0
        )
        task_relative.set_target(T_rel_target)

        ur5_tasks_with_constraint = ur5_tasks + [task_relative]

        box_trajectory = TrajectoryUtils.convert_dict_traj_to_mink_se3(big_box_trajectory)

        print("开始执行闭链运动...")
        execute_object_centric_trajectory(
            object_trajectory=box_trajectory,
            T_obj_to_left=T_obj_to_left,
            T_obj_to_right=T_obj_to_right,
            configuration=configuration,
            tasks=ur5_tasks_with_constraint,  # <--- 传入新列表
            limits=ur5_limits,
            solver=solver,
            model=model,
            data=data,
            viewer=viewer,
            rate=rate,
            arm_infos=ur5_arm_infos,
            gripper_vals=[1, 1]
        )
        print('静止在这个位置')
        open_gripper_position = box_trajectory[-1]

        wait_for_object_convergence(open_gripper_position, BigBox, T_obj_to_left, T_obj_to_right, configuration,
                                    ur5_tasks_with_constraint, ur5_limits, solver, model, data, viewer, rate,
                                    ur5_arm_infos, [1, 1], 10)
