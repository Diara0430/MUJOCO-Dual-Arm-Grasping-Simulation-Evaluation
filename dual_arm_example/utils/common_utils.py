import numpy as np
import mujoco
import mink
from scipy.spatial.transform import Rotation as R, Slerp


# ==========================================
# 数据格式校验与转换 (Ensure Helpers)
# ==========================================

class Format:
    """数据格式化与校验工具类"""

    @staticmethod
    def ensure_list(x):
        """确保输入被包裹在列表中"""
        if isinstance(x, list):
            return x
        return [x]

    @staticmethod
    def ensure_numpy_2d(x):
        """确保输入是 (N, 3) 或 (N, 3, 3) 的 Numpy 数组"""
        arr = np.array(x)
        if arr.ndim == 1:
            return arr[np.newaxis, :]
        return arr

    @staticmethod
    def ensure_rotation_batch(x):
        """专门处理旋转矩阵，确保输出是 (N, 3, 3)"""
        arr = np.array(x)
        if arr.ndim == 2:  # (3, 3)
            return arr[np.newaxis, :, :]
        return arr


class TrajectoryUtils:
    @staticmethod
    def get_current_pos_and_rot(mj_data, *targets):
        """
        获取指定目标 (RobotArm对象 或 site_id) 的当前笛卡尔位置和旋转矩阵。

        Args:
            mj_data: mujoco.MjData 对象
            *targets:
                - RobotArm 实例 (会自动读取其 .site_id)
                - int (直接传入 site_id)

        Returns:
            pos_arr: np.ndarray, shape (N, 3)
            rot_arr: np.ndarray, shape (N, 3, 3)
        """
        positions = []
        rotations = []

        for target in targets:
            # --- 关键修改开始 ---
            # 自动判断传入的是 RobotArm 对象还是纯 ID
            if hasattr(target, 'site_id'):
                sid = target.site_id
            else:
                sid = target
            # --- 关键修改结束 ---

            # 1. 获取位置
            pos = mj_data.site_xpos[sid].copy()
            positions.append(pos)

            # 2. 获取旋转矩阵
            rot_flat = mj_data.site_xmat[sid]
            rot_mat = rot_flat.reshape(3, 3).copy()
            rotations.append(rot_mat)

        return np.array(positions), np.array(rotations)

    @staticmethod
    def merge_trajectory_data(traj_list, pos_threshold=5e-4, rot_threshold=1e-3):
        """
        用于把获取的单个box轨迹的多个轨迹片段合并，并过滤掉重复或静止的点。

        Args:
            traj_list: 包含 {'pos', 'rot'} 的列表
            pos_threshold: 位置变化阈值 (米)，默认 0.01mm
            rot_threshold: 旋转变化阈值 (矩阵范数)，默认约 0.05度

        Returns:
            清理后的字典 {'pos': (M, 3), 'rot': (M, 3, 3)}
        """
        # 1. 过滤 None 并初步提取
        valid_trajs = [t for t in traj_list if t is not None]
        if not valid_trajs:
            return None

        all_pos = [t['pos'] for t in valid_trajs]
        all_rot = [t['rot'] for t in valid_trajs]

        # 2. 先暴力合并所有数据 (Concatenate)
        merged_pos = np.concatenate(all_pos, axis=0)
        merged_rot = np.concatenate(all_rot, axis=0)

        # 如果数据量太少（比如只有1帧），直接返回
        if len(merged_pos) <= 1:
            return {'pos': merged_pos, 'rot': merged_rot}

        # 3. 计算相邻帧的差异 (Vectorized Calculation)
        # pos_diff[i] 表示第 i+1 帧和第 i 帧的欧氏距离
        # shape: (N-1,)
        pos_diff = np.linalg.norm(merged_pos[1:] - merged_pos[:-1], axis=1)

        # rot_diff[i] 表示旋转矩阵差异的 Frobenius 范数
        # shape: (N-1,)
        rot_diff = np.linalg.norm(merged_rot[1:] - merged_rot[:-1], axis=(1, 2))

        # 4. 生成保留掩码 (Mask)
        # 保留条件：位置变化 > 阈值  或者  旋转变化 > 阈值
        is_moving = (pos_diff > pos_threshold) | (rot_diff > rot_threshold)

        # 注意：第一帧永远保留 (np.concatenate(([True], ...)))
        keep_mask = np.concatenate(([True], is_moving))

        # 5. 应用掩码，筛选数据
        clean_pos = merged_pos[keep_mask]
        clean_rot = merged_rot[keep_mask]

        # 打印一下优化效果（可选）
        print(f"轨迹合并优化: 原始 {len(merged_pos)} 帧 -> 优化后 {len(clean_pos)} 帧 "
              f"(去除了 {len(merged_pos) - len(clean_pos)} 个重复/静止点)")

        return {
            'pos': clean_pos,
            'rot': clean_rot
        }

    @staticmethod
    def transaction_trajectory_data_alignment(traj_data):
        pos = traj_data['pos']  # (N, 3)
        N = len(pos)
        start = pos[0]
        end = pos[-1]

        t = np.linspace(0, 1, N)
        linear = start + np.outer(t, (end - start))
        deviation = pos - linear

        # Heading Alignment (转到 Canonical X轴)
        vec = end - start
        vec_xy = vec[:2] / (np.linalg.norm(vec[:2]) + 1e-9)
        yaw = np.arctan2(vec_xy[1], vec_xy[0])

        r_inv = R.from_euler('z', -yaw)
        dev_canonical = r_inv.apply(deviation)
        return dev_canonical

    @staticmethod
    def rotation_trajectory_data_alignment(traj_data, single_arm_base_pos):
        """
        提取给过来的轨迹数据里面的旋转部分的细节数据，在原始旋转数据里去除因为起点和终点产生的被动旋转。并且把旋转数据转为李代数形式
        :param traj_data:
        :param single_arm_base_pos:
        :return:
        """
        ref_pos = traj_data['pos']
        ref_rot = traj_data['rot']
        N = len(ref_pos)

        relative_pos = ref_pos - single_arm_base_pos
        yaws = np.arctan2(relative_pos[:, 1], relative_pos[:, 0])
        base_rot = R.from_euler('z', yaws)

        obj_rot_in_world = R.from_matrix(ref_rot)
        obj_rot_in_local = base_rot.inv() * obj_rot_in_world

        continuous_vecs = np.zeros((N, 3))
        # 第一帧直接取值
        continuous_vecs[0] = obj_rot_in_local[0].as_rotvec()

        for i in range(1, N):
            # 计算 R[i-1] 到 R[i] 的微小旋转差
            # R_diff * R[i-1] = R[i]  =>  R_diff = R[i] * R[i-1].inv()
            # 注意：这里是在局部坐标系下的微小变化，右乘即可
            r_diff = obj_rot_in_local[i] * obj_rot_in_local[i - 1].inv()

            # 提取微小旋转向量 (这个值一定很小，不会跳变)
            diff_vec = r_diff.as_rotvec()

            # 累加
            continuous_vecs[i] = continuous_vecs[i - 1] + diff_vec

        return continuous_vecs

    @staticmethod
    def apply_deviation_2_new_line(new_start_pos, new_end_pos, promp_deviation, ref_N):
        """
        把通过promp 处理后的弧度轨迹 叠加到新起点和新终点生成的line上
        :param new_start_pos: 新的起点位置
        :param new_end_pos: 新的结束位置
        :param promp_deviation: 通过promp生成的轨迹偏差
        :param ref_N:
        :return:
        """
        vec_new = new_end_pos - new_start_pos
        vec_new_xy = vec_new[:2] / (np.linalg.norm(vec_new[:2]) + 1e-9)
        yaw_new = np.arctan2(vec_new_xy[1], vec_new_xy[0])

        # 2. 【关键】正向旋转，把标准偏差转到新方向
        r_new = R.from_euler('z', yaw_new)
        rotated_deviation = r_new.apply(promp_deviation)

        # 3. 叠加到新直线
        t_steps = np.linspace(0, 1, ref_N)
        new_linear = new_start_pos + np.outer(t_steps, (new_end_pos - new_start_pos))
        new_pos_traj = new_linear + rotated_deviation

        return new_pos_traj

    @staticmethod
    def apply_rot_details_2_new_base_rot(rot_details_promp, new_trans_traj, dual_arm_robot_base_pos, new_start_rot_mat):
        big_obj_pos_rel = new_trans_traj - dual_arm_robot_base_pos
        big_obj_yaws = np.arctan2(big_obj_pos_rel[:, 1], big_obj_pos_rel[:, 0])
        r_base_new = R.from_euler('z', big_obj_yaws)

        # 2. 合成最终旋转 (R_final = R_base_new * R_local_new)
        # 把 ProMP 生成的动作特征，安装到新的手臂方向上
        r_new_traj_raw = r_base_new * rot_details_promp

        # 3. 起点修正 (Global Correction)
        # 确保第一帧姿态严格对齐 big_start_rot_mat
        r_current_start = r_new_traj_raw[0]
        r_target_start = R.from_matrix(new_start_rot_mat)

        # 修正矩阵 (左乘)
        r_correction = r_target_start * r_current_start.inv()

        # 应用修正
        r_final_traj = r_correction * r_new_traj_raw

        return r_final_traj

    @staticmethod
    def convert_dict_traj_to_mink_se3(traj_dict):
        """
        将 {'pos': (N,3), 'rot': (N,3,3)} 转换为 List[mink.SE3]
        """
        pos_arr = traj_dict['pos']
        rot_arr = traj_dict['rot']

        # 使用 Scipy 批量转换旋转矩阵为四元数 (x, y, z, w)
        r_objs = R.from_matrix(rot_arr)
        quats_xyzw = r_objs.as_quat()

        se3_list = []
        for i in range(len(pos_arr)):
            pos = pos_arr[i]
            x, y, z, w = quats_xyzw[i]

            # 注意：Mink/MuJoCo 的四元数顺序通常是 wxyz (scalar first)
            # Scipy 是 xyzw (scalar last)，所以这里要调整顺序
            se3 = mink.SE3.from_rotation_and_translation(
                rotation=mink.SO3(wxyz=np.array([w, x, y, z])),
                translation=pos
            )
            se3_list.append(se3)

        return se3_list
