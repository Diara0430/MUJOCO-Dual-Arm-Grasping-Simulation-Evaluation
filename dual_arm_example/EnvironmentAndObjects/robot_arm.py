import mujoco
import mujoco.viewer
import numpy as np
import mink
from scipy.spatial.transform import Rotation as R


class RobotArm:
    def __init__(self, name, joint_names, attachment_site, root_actuator_name, gripper_actuator_name=None, home_q=None):
        self.name = name
        self.joint_names = joint_names
        self.attachment_site_name = attachment_site
        self.root_actuator_name = root_actuator_name
        self.gripper_actuator_name = gripper_actuator_name
        self.home_q = home_q

        self.site_id = -1
        self.qpos_idx = -1
        self.ctrl_idx = -1
        self.gripper_ctrl_idx = None
        self.dof = len(joint_names)

    def initialize(self, model):
        if self.attachment_site_name is not None:
            self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self.attachment_site_name)
            if self.site_id == -1:
                raise ValueError(f"[{self.name}] Site not found: {self.attachment_site_name}")

        first_joint_name = self.joint_names[0]
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, first_joint_name)
        if joint_id == -1: raise ValueError(f"[{self.name}] Joint not found: {first_joint_name}")
        self.qpos_idx = model.jnt_qposadr[joint_id]

        self.ctrl_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, self.root_actuator_name)
        if self.ctrl_idx == -1: raise ValueError(f"[{self.name}] Actuator not found: {self.root_actuator_name}")

        if self.gripper_actuator_name:
            self.gripper_ctrl_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, self.gripper_actuator_name)

    def get_control_info(self):
        return {
            'qpos_idx': self.qpos_idx,
            'ctrl_idx': self.ctrl_idx,
            'gripper_ctrl_idx': self.gripper_ctrl_idx,
            'range': self.dof
        }

    def get_xpos_and_rot(self, data):
        if self.site_id is None:
            raise ValueError(f"[{self.name}] Cannot get xpos: No site defined for this component.")

        xpos = data.site_xpos[self.site_id].copy()
        rotation = data.site_xmat[self.site_id].copy().reshape(3, 3)
        return xpos, rotation

    def get_se3(self, data):
        if self.site_id is None:
            raise ValueError(f"[{self.name}] Cannot get se3: No site defined for this component.")

        xpos = data.site_xpos[self.site_id].copy()
        rotation = data.site_xmat[self.site_id].copy().reshape(3, 3)
        r = R.from_matrix(rotation)
        x, y, z, w = r.as_quat()
        return mink.SE3.from_rotation_and_translation(
            rotation=mink.SO3(wxyz=np.array([w, x, y, z])),
            translation=xpos
        )

    def set_rotation_info(self, angle, axis='z'):
        return {'name': self.name, 'site_id': self.site_id, 'angle': angle, 'axis': axis}

    def reset_to_home(self, data):
        if self.home_q is not None:
            data.qpos[self.qpos_idx: self.qpos_idx + self.dof] = self.home_q