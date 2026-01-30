import mujoco
import mujoco.viewer
import numpy as np
import mink
from scipy.spatial.transform import Rotation as R, Slerp
from utils.transfer_from_small_2_big import affine_transform
from utils.common_utils import Format


class SceneObject:
    def __init__(self, body_name, model, data, geom_name=None):
        """
        环境物体类，用于管理物体的状态、特征点及与其他物体的变换关系。

        Args:
            name (str): Body 的名称 (xml中的 name)
            model: mujoco.MjModel
            data: mujoco.MjData
            geom_name (str, optional): 如果需要获取物体的几何特征点(如角点)，传入对应的 geom name
        """
        self.body_name = body_name
        self.model = model
        self.data = data
        self.geom_name = geom_name

        # 获取 Body ID
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.body_name)
        if self.body_id == -1:
            raise ValueError(f"[SceneObject] Body not found: {self.body_name}")

        # 存储仿射变换矩阵 (A_aff, t_aff)，用于将本物体上的点映射到目标物体
        self.A_aff = None
        self.t_aff = None
        self.target_object_name = None

    def get_xpos_and_rot(self):
        """获取当前世界坐标系下的位置和旋转矩阵"""
        pos = self.data.xpos[self.body_id].copy()
        rot = self.data.xmat[self.body_id].copy().reshape(3, 3)
        return pos, rot

    def get_se3(self):
        """获取当前 mink.SE3 位姿"""
        pos, rot = self.get_xpos_and_rot()
        r = R.from_matrix(rot)
        x, y, z, w = r.as_quat()
        return mink.SE3.from_rotation_and_translation(
            rotation=mink.SO3(wxyz=np.array([w, x, y, z])),
            translation=pos
        )

    def get_feature_points(self):
        """获取物体的关键特征点 (依赖 import 的 get_box_point)"""
        if self.geom_name is None:
            print(f"[{self.name}] Warning: No geom_name provided, cannot get feature points.")
            return None

        # 1. 获取 Geom ID
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, self.geom_name)
        if geom_id == -1:
            raise ValueError(f"Geom name '{self.geom_name}' not found.")

        # 2. 获取 Geom 的尺寸 (hx, hy, hz) -> 对应 XML 中的 size="hx hy hz"
        # 注意：box 的 size 是半长 (half-size)
        size = self.model.geom_size[geom_id]
        hx, hy, hz = size

        # 3. 定义局部坐标系下的 8 个顶点
        box_vertices_local = np.array([
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz]
        ])

        # 4. 获取 Geom 在世界坐标系下的位置和旋转矩阵
        # 关键修改：使用 geom_xpos 和 geom_xmat，而不是 body 的 xpos/xmat
        geom_xpos = self.data.geom_xpos[geom_id]
        geom_xmat = self.data.geom_xmat[geom_id].reshape(3, 3)

        # 5. 坐标变换：局部 -> 世界
        # P_world = R * P_local + T
        box_vertices_world = (geom_xmat @ box_vertices_local.T).T + geom_xpos

        return box_vertices_world

    def compute_affine_transform_to(self, target_obj):
        """
        计算从本物体(Source)到目标物体(Target)的仿射变换矩阵。

        Args:
            target_obj (SceneObject): 目标物体实例
        """
        src_points = self.get_feature_points()
        tgt_points = target_obj.get_feature_points()

        if src_points is None or tgt_points is None:
            raise ValueError("无法计算仿射变换：源物体或目标物体无法获取特征点。")

        # 调用你原来导入的 affine_transform
        # A_aff: (3,3), t_aff: (3,)
        self.A_aff, self.t_aff, _ = affine_transform(src_points, tgt_points)
        self.target_object_name = target_obj.body_name
        print(f"[{self.body_name}] -> [{target_obj.body_name}] 仿射变换矩阵已计算。")

        return self.A_aff, self.t_aff

    def map_points_to_target(self, points):
        """
        将本物体上的点（如抓取点）映射到目标物体坐标系中。
        公式: points_new = points @ A.T + t
        """
        if self.A_aff is None or self.t_aff is None:
            raise RuntimeError(f"[{self.body_name}] 尚未计算仿射变换，请先调用 compute_affine_transform_to。")

        points = Format.ensure_numpy_2d(points)
        return points @ self.A_aff.T + self.t_aff
