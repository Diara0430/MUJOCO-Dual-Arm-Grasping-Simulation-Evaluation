import csv
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import traceback
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter
import mink

from EnvironmentAndObjects.robot_arm import RobotArm


# =========================
# 配置区
# =========================
LEFT_EE_SITE  = "ur_grip_site_left"
RIGHT_EE_SITE = "ur_grip_site_right"

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

LEFT_FINGERTIP_GEOMS  = ["left_fingertip_visual_ur5left", "right_fingertip_visual_ur5left"]
RIGHT_FINGERTIP_GEOMS = ["left_fingertip_visual_ur5right", "right_fingertip_visual_ur5right"]

NUM_CANDIDATES  = 100
PREGRASP_OFFSET = 0.10
LIFT_HEIGHT     = 0.30
HOLD_TIME       = 2.0

CONTROL_FREQ = 60.0
SOLVER = "daqp"
MAX_VEL = np.pi

GRIPPER_OPEN  = -1.0
GRIPPER_CLOSE =  1.0

# marker 可视化参数
MARKER_SPHERE_R  = 0.012
MARKER_ARROW_W   = 0.003
MARKER_ARROW_LEN = 0.08

# ✅ 稳定性/鲁棒性
ALLOW_HEADLESS_AFTER_VIEWER_CLOSE = True
CONTINUE_ON_CANDIDATE_EXCEPTION = True

# ✅ 打印模式：A（阶段开始/结束 + 最终汇总）
VERBOSE_STAGE = True

# ✅ 接触力统计（可选）
ENABLE_CONTACT_FORCE_STATS = True

np.random.seed(0)


# =========================
# 键盘控制状态（q/space/./n/v）
# =========================
ctrl_state = {
    "quit": False,      # q: 退出
    "paused": False,    # space: 暂停/继续
    "step_once": False, # .: 单步
    "skip": False,      # n: 跳过 candidate
    "show": False,      # v: 显示/隐藏 marker
}
state_lock = threading.Lock()

def key_callback(keycode: int):
    try:
        k = chr(keycode).lower()
    except Exception:
        return
    with state_lock:
        if k == 'q':
            ctrl_state["quit"] = True
        elif k == ' ':
            ctrl_state["paused"] = not ctrl_state["paused"]
        elif k == '.':
            ctrl_state["paused"] = True
            ctrl_state["step_once"] = True
        elif k == 'n':
            ctrl_state["skip"] = True
        elif k == 'v':
            ctrl_state["show"] = not ctrl_state["show"]

def should_quit_or_skip():
    with state_lock:
        return ctrl_state["quit"], ctrl_state["skip"], ctrl_state["paused"], ctrl_state["step_once"]

def consume_step_once():
    with state_lock:
        ctrl_state["step_once"] = False


# =========================
# 打印工具（模仿“开始/完成/警告/最终”风格）
# =========================
def _fmt_arr(x, nd=4):
    a = np.array(x).reshape(-1)
    return "[" + ", ".join([f"{v:.{nd}f}" for v in a]) + "]"

def log_stage(msg: str):
    if VERBOSE_STAGE:
        print(msg)

def log_kv(prefix: str, **kwargs):
    if not VERBOSE_STAGE:
        return
    parts = []
    for k, v in kwargs.items():
        if isinstance(v, (list, tuple, np.ndarray)):
            parts.append(f"{k}={_fmt_arr(v)}")
        else:
            parts.append(f"{k}={v}")
    print(prefix + " " + " | ".join(parts))


# =========================
# 数据结构（世界/中心/物体坐标）
# =========================
@dataclass
class GraspCandidate:
    # world frame
    left_pos_w: np.ndarray
    right_pos_w: np.ndarray
    left_rot_w: np.ndarray
    right_rot_w: np.ndarray

    # relative to AABB geometric center in world (translation-only)
    left_pos_center: np.ndarray
    right_pos_center: np.ndarray

    # object/body frame
    left_pos_obj: np.ndarray
    right_pos_obj: np.ndarray
    left_rot_obj: np.ndarray
    right_rot_obj: np.ndarray


# =========================
# Logger
# =========================
class ResultLogger:
    def __init__(self, out_dir="results"):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.root = Path(out_dir) / ts
        self.root.mkdir(parents=True, exist_ok=True)

    def open_csv(self, name, fieldnames):
        path = self.root / f"{name}.csv"
        f = open(path, "w", newline="", encoding="utf-8")
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        return f, w, path

    def save_json(self, name, obj):
        path = self.root / f"{name}.json"
        path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
        return path


# =========================
# Mink / MuJoCo 工具函数
# =========================
def to_mink_se3(pos, rot):
    quat_xyzw = R.from_matrix(rot).as_quat()  # x y z w
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    return mink.SE3.from_rotation_and_translation(
        rotation=mink.SO3(wxyz=quat_wxyz),
        translation=pos
    )

def make_rotation_from_approach(approach_dir, up_hint=np.array([0, 0, 1.0])):
    z = approach_dir / (np.linalg.norm(approach_dir) + 1e-9)
    x = np.cross(up_hint, z)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(np.array([0, 1.0, 0]), z)
    x = x / (np.linalg.norm(x) + 1e-9)
    y = np.cross(z, x)
    y = y / (np.linalg.norm(y) + 1e-9)
    return np.column_stack([x, y, z])

def world_to_obj(p_w, obj_pos_w, obj_rot_w):
    return obj_rot_w.T @ (p_w - obj_pos_w)

def obj_to_world(p_obj, obj_pos_w, obj_rot_w):
    return obj_pos_w + (obj_rot_w @ p_obj)

def rot_world_to_obj(R_w, obj_rot_w):
    return obj_rot_w.T @ R_w

def rot_obj_to_world(R_obj_local, obj_rot_w):
    return obj_rot_w @ R_obj_local

def get_geom_ids(model, names):
    ids = []
    for n in names:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
        if gid == -1:
            raise ValueError(f"Geom not found: {n}")
        ids.append(gid)
    return ids

def contact_exists(data, geom_ids_a, geom_ids_b):
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = c.geom1, c.geom2
        if (g1 in geom_ids_a and g2 in geom_ids_b) or (g2 in geom_ids_a and g1 in geom_ids_b):
            return True
    return False

def min_contact_dist_between(data, geom_ids_a, geom_ids_b):
    dmin = 1e9
    found = False
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = c.geom1, c.geom2
        if (g1 in geom_ids_a and g2 in geom_ids_b) or (g2 in geom_ids_a and g1 in geom_ids_b):
            found = True
            dmin = min(dmin, float(c.dist))
    return (dmin if found else 999.0), found

def mean_contact_force_norm(model, data, geom_ids_a, geom_ids_b):
    """
    简化统计：对 fingertips-object 的 contact，取 mj_contactForce 的 force(3) 范数平均。
    """
    if not ENABLE_CONTACT_FORCE_STATS:
        return 0.0, 0
    total = 0.0
    cnt = 0
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = c.geom1, c.geom2
        if not ((g1 in geom_ids_a and g2 in geom_ids_b) or (g2 in geom_ids_a and g1 in geom_ids_b)):
            continue
        res = np.zeros(6, dtype=np.float64)
        try:
            mujoco.mj_contactForce(model, data, i, res)
            fn = float(np.linalg.norm(res[:3]))
            total += fn
            cnt += 1
        except Exception:
            pass
    return (total / cnt if cnt > 0 else 0.0), cnt

def init_tasks(configuration, model, joint_names, ee_sites, max_vel=MAX_VEL):
    max_velocities = {name: max_vel for name in joint_names}
    limits = [
        mink.ConfigurationLimit(model=configuration.model),
        mink.VelocityLimit(model, max_velocities),
    ]
    tasks = []
    for site in ee_sites:
        tasks.append(mink.FrameTask(
            frame_name=site, frame_type="site",
            position_cost=10.0, orientation_cost=1.0, lm_damping=1.0
        ))
    posture = mink.PostureTask(model, cost=1e-3)
    posture.set_target(configuration.q)
    tasks.append(posture)
    return tasks, limits

def se3_rel_error(T_rel_now: mink.SE3, T_rel_target: mink.SE3):
    dp = T_rel_now.translation() - T_rel_target.translation()
    pos_err = float(np.linalg.norm(dp))
    R_now = T_rel_now.rotation().as_matrix()
    R_tar = T_rel_target.rotation().as_matrix()
    R_diff = R.from_matrix(R_tar) * R.from_matrix(R_now).inv()
    rot_err_deg = float(np.rad2deg(R_diff.magnitude()))
    return pos_err, rot_err_deg

def get_obj_pose(model, data, obj_body_id):
    mujoco.mj_forward(model, data)
    pos = data.xpos[obj_body_id].copy()
    rot = data.xmat[obj_body_id].copy().reshape(3,3)
    return pos, rot


# =========================
# Viewer 安全调用
# =========================
def viewer_is_running(viewer):
    if viewer is None:
        return False
    try:
        return viewer.is_running()
    except Exception:
        return False

def viewer_safe_sync(viewer):
    if viewer is None:
        return
    try:
        viewer.sync()
    except Exception:
        pass

def viewer_safe_close(viewer):
    if viewer is None:
        return
    try:
        viewer.close()
    except Exception:
        pass


# =========================
# Marker 可视化（球+箭头）—兼容 MuJoCo 2.3/3.x，且贴物体
# =========================
def clear_markers(viewer):
    if viewer is None:
        return
    try:
        viewer.user_scn.ngeom = 0
    except Exception:
        pass

def add_sphere(viewer, pos, radius=0.01, rgba=(1, 0, 0, 0.9)):
    if viewer is None:
        return
    scn = viewer.user_scn
    i = scn.ngeom
    if i >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(
        scn.geoms[i],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0, 0], dtype=float),
        np.array(pos, dtype=float),
        np.eye(3).reshape(-1),
        np.array(rgba, dtype=float)
    )
    scn.geoms[i].dataid = -1
    scn.ngeom += 1

def _mjv_make_arrow_geom(geom, width, start, end):
    # MuJoCo 2.3: mjv_makeConnector; MuJoCo 3.x: mjv_connector
    try:
        mujoco.mjv_makeConnector(
            geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            width,
            float(start[0]), float(start[1]), float(start[2]),
            float(end[0]),   float(end[1]),   float(end[2]),
        )
    except AttributeError:
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            width,
            float(start[0]), float(start[1]), float(start[2]),
            float(end[0]),   float(end[1]),   float(end[2]),
        )

def add_arrow(viewer, start, end, rgba=(0, 1, 0, 0.9), width=0.003):
    if viewer is None:
        return
    scn = viewer.user_scn
    i = scn.ngeom
    if i >= scn.maxgeom:
        return
    _mjv_make_arrow_geom(scn.geoms[i], width, start, end)
    scn.geoms[i].rgba[:] = np.array(rgba, dtype=float)
    scn.geoms[i].dataid = -1
    scn.ngeom += 1

def show_grasp_pair_markers_on_object(viewer, cand: GraspCandidate, obj_pos_w, obj_rot_w):
    """cand 存物体系局部抓取点 -> 投回当前世界，marker 永远贴着物体"""
    if viewer is None:
        return
    clear_markers(viewer)

    Lw = obj_to_world(cand.left_pos_obj,  obj_pos_w, obj_rot_w)
    Rw = obj_to_world(cand.right_pos_obj, obj_pos_w, obj_rot_w)

    add_sphere(viewer, Lw, radius=MARKER_SPHERE_R, rgba=(1,0,0,0.9))
    add_sphere(viewer, Rw, radius=MARKER_SPHERE_R, rgba=(0,0,1,0.9))

    Lrot_w = rot_obj_to_world(cand.left_rot_obj,  obj_rot_w)
    Rrot_w = rot_obj_to_world(cand.right_rot_obj, obj_rot_w)

    aL = Lrot_w[:, 2]
    aR = Rrot_w[:, 2]
    add_arrow(viewer, Lw, Lw + MARKER_ARROW_LEN * aL, rgba=(1,0.3,0.3,0.9), width=MARKER_ARROW_W)
    add_arrow(viewer, Rw, Rw + MARKER_ARROW_LEN * aR, rgba=(0.3,0.3,1,0.9), width=MARKER_ARROW_W)


# =========================
# stepping
# =========================
def step_ik_and_sim(model, data, configuration, tasks, limits, solver, rate,
                    arm_infos, gripper_val, physics_steps):
    quit_flag, skip_flag, paused, step_once = should_quit_or_skip()
    if quit_flag:
        return "quit"
    if skip_flag:
        return "skip"

    configuration.update(data.qpos)
    vel = mink.solve_ik(configuration, tasks, rate.dt, solver, 1e-3, limits=limits)
    configuration.integrate_inplace(vel, rate.dt)

    for info in arm_infos:
        c_idx, q_idx, dof = info['ctrl_idx'], info['qpos_idx'], info['range']
        data.ctrl[c_idx:c_idx+dof] = configuration.q[q_idx:q_idx+dof]
        if info.get('gripper_ctrl_idx', None) is not None:
            data.ctrl[info['gripper_ctrl_idx']] = gripper_val

    if paused and (not step_once):
        rate.sleep()
        return "paused"

    if step_once:
        consume_step_once()

    for _ in range(physics_steps):
        for info in arm_infos:
            c_idx, dof = info['ctrl_idx'], info['range']
            data.qfrc_applied[c_idx:c_idx+dof] = data.qfrc_bias[c_idx:c_idx+dof]
        mujoco.mj_step(model, data)

    rate.sleep()
    return "ok"


# =========================
# state utils
# =========================
def save_state_full(model, data):
    sz = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    buf = np.zeros(sz, dtype=np.float64)
    mujoco.mj_getState(model, data, buf, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    return buf

def load_state_full(model, data, buf):
    mujoco.mj_setState(model, data, buf, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    data.ctrl[:] = 0
    if hasattr(data, "act") and data.act is not None:
        data.act[:] = 0
    data.qfrc_applied[:] = 0
    mujoco.mj_forward(model, data)

def settle_free_fall(model, data, steps=500):
    data.ctrl[:] = 0
    if hasattr(data, "act") and data.act is not None:
        data.act[:] = 0
    data.qfrc_applied[:] = 0
    for _ in range(steps):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)


# =========================
# scene discovery
# =========================
def discover_freejoint_object(model):
    free_joint_ids = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    if not free_joint_ids:
        raise ValueError("No freejoint found in scene. Scheme B expects exactly one freejoint object.")
    j0 = free_joint_ids[0]
    body_id = model.jnt_bodyid[j0]
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)

    geom_ids = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body_id]
    primary = None
    for g in geom_ids:
        if model.geom_contype[g] != 0 and model.geom_conaffinity[g] != 0:
            primary = g
            break
    if primary is None and geom_ids:
        primary = geom_ids[0]

    geom_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in geom_ids]
    return body_id, body_name, geom_ids, geom_names, primary


# =========================
# AABB + candidate sampling
# =========================
def get_object_aabb_world(model, data, obj_geom_id):
    mujoco.mj_forward(model, data)
    gpos = data.geom_xpos[obj_geom_id].copy()
    gmat = data.geom_xmat[obj_geom_id].copy().reshape(3,3)

    if model.geom_type[obj_geom_id] == mujoco.mjtGeom.mjGEOM_BOX:
        hx, hy, hz = model.geom_size[obj_geom_id]
        corners_local = np.array([
            [ hx,  hy,  hz],
            [ hx,  hy, -hz],
            [ hx, -hy,  hz],
            [ hx, -hy, -hz],
            [-hx,  hy,  hz],
            [-hx,  hy, -hz],
            [-hx, -hy,  hz],
            [-hx, -hy, -hz],
        ])
        corners_world = (gmat @ corners_local.T).T + gpos
        mins = corners_world.min(axis=0)
        maxs = corners_world.max(axis=0)
        center = (mins + maxs) / 2.0
        extent = (maxs - mins)
        return mins, maxs, center, extent

    mins = gpos - 0.05
    maxs = gpos + 0.05
    center = gpos
    extent = maxs - mins
    return mins, maxs, center, extent


def build_grasp_candidates_adaptive(mins, maxs, center, extent,
                                   obj_pos_w, obj_rot_w,
                                   num_candidates=100,
                                   surface_eps=0.006, z_margin=0.004):
    xy_span = float(np.linalg.norm(extent[:2]) + 1e-9)
    min_pair_dist = 0.5 * xy_span
    max_pair_dist = 1.5 * xy_span

    candidates = []
    tries = 0

    def sample_one():
        face = np.random.randint(0, 5)  # 不采样底面
        u, v = np.random.rand(), np.random.rand()
        p = np.array([
            mins[0] + u*(maxs[0]-mins[0]),
            mins[1] + v*(maxs[1]-mins[1]),
            mins[2] + np.random.rand()*(maxs[2]-mins[2]),
        ])
        n = np.zeros(3)
        if face == 0: p[0] = mins[0]; n = np.array([-1,0,0])
        if face == 1: p[0] = maxs[0]; n = np.array([ 1,0,0])
        if face == 2: p[1] = mins[1]; n = np.array([0,-1,0])
        if face == 3: p[1] = maxs[1]; n = np.array([0, 1,0])
        if face == 4: p[2] = maxs[2]; n = np.array([0, 0,1])

        if p[2] < mins[2] + z_margin or p[2] > maxs[2] - z_margin:
            return None
        p = p + surface_eps * n
        return p

    while len(candidates) < num_candidates and tries < num_candidates * 800:
        tries += 1
        p1 = sample_one()
        p2 = sample_one()
        if p1 is None or p2 is None:
            continue

        d = np.linalg.norm(p1 - p2)
        if d < min_pair_dist or d > max_pair_dist:
            continue

        if p1[1] <= p2[1]:
            left_p_w, right_p_w = p1, p2
        else:
            left_p_w, right_p_w = p2, p1

        left_rot_w  = make_rotation_from_approach(center - left_p_w)
        right_rot_w = make_rotation_from_approach(center - right_p_w)

        left_center  = left_p_w - center
        right_center = right_p_w - center

        left_p_obj  = world_to_obj(left_p_w,  obj_pos_w, obj_rot_w)
        right_p_obj = world_to_obj(right_p_w, obj_pos_w, obj_rot_w)

        left_R_obj  = rot_world_to_obj(left_rot_w,  obj_rot_w)
        right_R_obj = rot_world_to_obj(right_rot_w, obj_rot_w)

        candidates.append(GraspCandidate(
            left_pos_w=left_p_w, right_pos_w=right_p_w,
            left_rot_w=left_rot_w, right_rot_w=right_rot_w,
            left_pos_center=left_center, right_pos_center=right_center,
            left_pos_obj=left_p_obj, right_pos_obj=right_p_obj,
            left_rot_obj=left_R_obj, right_rot_obj=right_R_obj
        ))

    return candidates


def calc_slip(obj_se3, left_se3, right_se3, T_obj_to_left_init, T_obj_to_right_init):
    T_obj_to_left_now = obj_se3.inverse().multiply(left_se3)
    T_obj_to_right_now = obj_se3.inverse().multiply(right_se3)
    dL = np.linalg.norm(T_obj_to_left_now.translation() - T_obj_to_left_init.translation())
    dR = np.linalg.norm(T_obj_to_right_now.translation() - T_obj_to_right_init.translation())
    return float(dL), float(dR)


# =========================
# Stage CSV helpers (固定 5 行/ candidate)
# =========================
STAGE_LIST = [
    (1, "Stage1_pregrasp"),
    (2, "Stage2_grasp"),
    (3, "Stage3_close"),
    (4, "Stage4_lift"),
    (5, "Stage5_hold"),
]

STAGE_FIELDNAMES = [
    "scene","object_body","candidate_id",
    "stage_idx","stage","stage_ok","fail_reason",
    "sim_time",
    "obj_x","obj_y","obj_z",
    "lift_height","hold_duration","max_slip",
    "obj_v_rms","obj_w_rms","contact_ratio","min_contact_dist",
    "rel_pos_err","rel_rot_err_deg",
    "contact_force_n_mean","contact_force_n_count"
]

def make_stage_row_defaults():
    return {
        "lift_height": np.nan,
        "hold_duration": np.nan,
        "max_slip": np.nan,
        "obj_v_rms": np.nan,
        "obj_w_rms": np.nan,
        "contact_ratio": np.nan,
        "min_contact_dist": np.nan,
        "rel_pos_err": np.nan,
        "rel_rot_err_deg": np.nan,
        "contact_force_n_mean": np.nan,
        "contact_force_n_count": 0
    }

def write_stage_row(stage_w, stage_f,
                    scene, obj_name, cand_id,
                    stage_idx, stage_name,
                    stage_ok, fail_reason,
                    model, data, obj_body_id,
                    fields: dict | None = None):
    if stage_w is None:
        return
    mujoco.mj_forward(model, data)
    obj_pos = data.xpos[obj_body_id].copy()
    row = {
        "scene": scene,
        "object_body": obj_name,
        "candidate_id": cand_id,
        "stage_idx": int(stage_idx),
        "stage": stage_name,
        "stage_ok": int(stage_ok),
        "fail_reason": fail_reason,
        "sim_time": float(data.time),
        "obj_x": float(obj_pos[0]),
        "obj_y": float(obj_pos[1]),
        "obj_z": float(obj_pos[2]),
    }
    row.update(make_stage_row_defaults())
    if fields:
        row.update(fields)
    stage_w.writerow(row)
    stage_f.flush()

def write_skipped_stages(stage_w, stage_f,
                         scene, obj_name, cand_id,
                         failed_stage_idx, failed_stage_name, reason,
                         model, data, obj_body_id):
    """失败后补齐剩余阶段：skipped_due_to_failure(...)"""
    for idx, name in STAGE_LIST:
        if idx <= failed_stage_idx:
            continue
        write_stage_row(stage_w, stage_f, scene, obj_name, cand_id,
                        idx, name, False,
                        f"skipped_due_to_failure({failed_stage_name}:{reason})",
                        model, data, obj_body_id, fields=None)


# =========================
# evaluate candidate（阶段化评估 + A打印 + stage固定5行）
# =========================
def evaluate_candidate(candidate: GraspCandidate,
                       scene_name: str,
                       obj_body_name: str,
                       cand_id: int,
                       model, data, configuration, tasks, limits, solver,
                       viewer, rate, physics_steps,
                       ur5LeftArm, ur5RightArm, arm_infos,
                       obj_body_id, obj_geom_ids,
                       fingertip_geom_ids,
                       stage_w=None, stage_f=None):

    def final_fail_dict(fail_stage, reason, has_contact=0, lift_height=0.0, hold_duration=0.0, max_slip=999.0,
                        obj_v_rms=999.0, obj_w_rms=999.0, contact_ratio=0.0, min_contact_dist=999.0,
                        rel_pos_err=999.0, rel_rot_err_deg=999.0,
                        contact_force_n_mean=0.0, contact_force_n_count=0):
        return dict(
            success=0,
            has_contact=int(has_contact),
            lift_height=float(lift_height),
            hold_duration=float(hold_duration),
            max_slip=float(max_slip),
            obj_v_rms=float(obj_v_rms),
            obj_w_rms=float(obj_w_rms),
            contact_ratio=float(contact_ratio),
            min_contact_dist=float(min_contact_dist),
            rel_pos_err=float(rel_pos_err),
            rel_rot_err_deg=float(rel_rot_err_deg),
            contact_force_n_mean=float(contact_force_n_mean),
            contact_force_n_count=int(contact_force_n_count),
            fail_stage=fail_stage,
            fail_reason=reason
        )

    allow_headless = ALLOW_HEADLESS_AFTER_VIEWER_CLOSE

    # Candidate header prints
    log_stage(f"\n========== 开始评估 Candidate {cand_id} ==========")
    log_kv("抓取点(世界)", L=candidate.left_pos_w, R=candidate.right_pos_w)
    log_kv("抓取点(相对几何中心)", L=candidate.left_pos_center, R=candidate.right_pos_center)
    log_kv("抓取点(物体坐标系)", L=candidate.left_pos_obj, R=candidate.right_pos_obj)

    mujoco.mj_forward(model, data)
    z0 = float(data.xpos[obj_body_id][2])

    # pregrasp/grasp targets (world)
    left_approach = candidate.left_rot_w[:, 2]
    right_approach = candidate.right_rot_w[:, 2]
    left_pre = candidate.left_pos_w - PREGRASP_OFFSET * left_approach
    right_pre = candidate.right_pos_w - PREGRASP_OFFSET * right_approach

    T_left_pre = to_mink_se3(left_pre, candidate.left_rot_w)
    T_right_pre = to_mink_se3(right_pre, candidate.right_rot_w)
    T_left_grasp = to_mink_se3(candidate.left_pos_w, candidate.left_rot_w)
    T_right_grasp = to_mink_se3(candidate.right_pos_w, candidate.right_rot_w)

    def maybe_update_markers():
        with state_lock:
            show = ctrl_state["show"]
        if show and viewer is not None:
            obj_pos_now, obj_rot_now = get_obj_pose(model, data, obj_body_id)
            try:
                show_grasp_pair_markers_on_object(viewer, candidate, obj_pos_now, obj_rot_now)
            except Exception:
                pass

    # ---------- Stage 1: pregrasp ----------
    idx, stage_name = STAGE_LIST[0]
    log_stage("开始 Stage 1: pregrasp，移动到预抓取位姿...")
    log_kv("预抓取目标(世界)", left_pre=left_pre, right_pre=right_pre)
    try:
        for _ in range(int(0.6 / rate.dt)):
            if (viewer is not None) and (not viewer_is_running(viewer)):
                if allow_headless:
                    viewer = None
                else:
                    write_stage_row(stage_w, stage_f, scene_name, obj_body_name, cand_id,
                                    idx, stage_name, False, "viewer_closed",
                                    model, data, obj_body_id, fields=None)
                    write_skipped_stages(stage_w, stage_f, scene_name, obj_body_name, cand_id,
                                         idx, stage_name, "viewer_closed",
                                         model, data, obj_body_id)
                    log_stage("Stage 1 失败：viewer_closed")
                    return final_fail_dict(stage_name, "viewer_closed")

            tasks[0].set_target(T_left_pre)
            tasks[1].set_target(T_right_pre)
            st = step_ik_and_sim(model, data, configuration, tasks, limits, solver, rate,
                                 arm_infos, GRIPPER_OPEN, physics_steps)
            maybe_update_markers()
            viewer_safe_sync(viewer)
            if st in ("quit", "skip"):
                write_stage_row(stage_w, stage_f, scene_name, obj_body_name, cand_id,
                                idx, stage_name, False, st, model, data, obj_body_id, fields=None)
                write_skipped_stages(stage_w, stage_f, scene_name, obj_body_name, cand_id,
                                     idx, stage_name, st, model, data, obj_body_id)
                log_stage(f"Stage 1 失败：{st}")
                return final_fail_dict(stage_name, st)

        write_stage_row(stage_w, stage_f, scene_name, obj_body_name, cand_id,
                        idx, stage_name, True, "", model, data, obj_body_id, fields=None)
        log_stage("Stage 1 完成。")
    except Exception as e:
        reason = f"ik_fail:{type(e).__name__}"
        write_stage_row(stage_w, stage_f, scene_name, obj_body_name, cand_id,
                        idx, stage_name, False, reason, model, data, obj_body_id, fields=None)
        write_skipped_stages(stage_w, stage_f, scene_name, obj_body_name, cand_id,
                             idx, stage_name, reason, model, data, obj_body_id)
        log_stage(f"Stage 1 失败：{reason}")
        return final_fail_dict(stage_name, reason)

    # ---------- Stage 2: grasp ----------
    idx, stage_name = STAGE_LIST[1]
    log_stage("开始 Stage 2: grasp，移动到抓取位姿...")
    log_kv("抓取目标(世界)", left=candidate.left_pos_w, right=candidate.right_pos_w)
    try:
        for _ in range(int(0.8 / rate.dt)):
            if (viewer is not None) and (not viewer_is_running(viewer)):
                if allow_headless:
                    viewer = None
                else:
                    write_stage_row(stage_w, stage_f, scene_name, obj_body_name, cand_id,
                                    idx, stage_name, False, "viewer_closed",
                                    model, data, obj_body_id, fields=None)
                    write_skipped_stages(stage_w, stage_f, scene_name, obj_body_name, cand_id,
                                         idx, stage_name, "viewer_closed",
                                         model, data, obj_body_id)
                    log_stage("Stage 2 失败：viewer_closed")
                    return final_fail_dict(stage_name, "viewer_closed")

            tasks[0].set_target(T_left_grasp)
            tasks[1].set_target(T_right_grasp)
            st = step_ik_and_sim(model, data, configuration, tasks, limits, solver, rate,
                                 arm_infos, GRIPPER_OPEN, physics_steps)
            maybe_update_markers()
            viewer_safe_sync(viewer)
            if st in ("quit", "skip"):
                write_stage_row(stage_w, stage_f, scene_name, obj_body_name, cand_id,
                                idx, stage_name, False, st,
                                model, data, obj_body_id, fields=None)
                write_skipped_stages(stage_w, stage_f, scene_name, obj_body_name, cand_id,
                                     idx, stage_name, st,
                                     model, data, obj_body_id)
                log_stage(f"Stage 2 失败：{st}")
                return final_fail_dict(stage_name, st)

        write_stage_row(stage_w, stage_f, scene_name, obj_body_name, cand_id,
                        idx, stage_name, True, "",
                        model, data, obj_body_id, fields=None)
        log_stage("Stage 2 完成。")
    except Exception as e:
        reason = f"ik_fail:{type(e).__name__}"
        write_stage_row(stage_w, stage_f, scene_name, obj_body_name, cand_id,
                        idx, stage_name, False, reason,
                        model, data, obj_body_id, fields=None)
        write_skipped_stages(stage_w, stage_f, scene_name, obj_body_name, cand_id,
                             idx, stage_name, reason, model, data, obj_body_id)
        log_stage(f"Stage 2 失败：{reason}")
        return final_fail_dict(stage_name, reason)

    # ---------- Stage 3: close + contact ----------
    idx, stage_name = STAGE_LIST[2]
    log_stage("开始 Stage 3: close，关闭夹爪并检测接触...")
    try:
        for _ in range(int(0.5 / rate.dt)):
            if (viewer is not None) and (not viewer_is_running(viewer)):
                if allow_headless:
                    viewer = None
                else:
                    write_stage_row(stage_w, stage_f, scene_name, obj_body_name, cand_id,
                                    idx, stage_name, False, "viewer_closed",
                                    model, data, obj_body_id, fields=None)
                    write_skipped_stages(stage_w, stage_f, scene_name, obj_body_name, cand_id,
                                         idx, stage_name, "viewer_closed",
                                         model, data, obj_body_id)
                    log_stage("Stage 3 失败：viewer_closed")
                    return final_fail_dict(stage_name, "viewer_closed")

            tasks[0].set_target(T_left_grasp)
            tasks[1].set_target(T_right_grasp)
            st = step_ik_and_sim(model, data, configuration, tasks, limits, solver, rate,
                                 arm_infos, GRIPPER_CLOSE, physics_steps)
            maybe_update_markers()
            viewer_safe_sync(viewer)
            if st in ("quit", "skip"):
                write_stage_row(stage_w, stage_f, scene_name, obj_body_name, cand_id,
                                idx, stage_name, False, st, model, data, obj_body_id, fields=None)
                write_skipped_stages(stage_w, stage_f, scene_name, obj_body_name, cand_id,
                                     idx, stage_name, st, model, data, obj_body_id)
                log_stage(f"Stage 3 失败：{st}")
                return final_fail_dict(stage_name, st)

        mujoco.mj_forward(model, data)
        has_contact = contact_exists(data, fingertip_geom_ids, obj_geom_ids)
        min_cdist, _ = min_contact_dist_between(data, fingertip_geom_ids, obj_geom_ids)
        fn_mean, fn_cnt = mean_contact_force_norm(model, data, fingertip_geom_ids, obj_geom_ids)

        if not has_contact:
            reason = "no_contact"
            write_stage_row(stage_w, stage_f, scene_name, obj_body_name, cand_id,
                            idx, stage_name, False, reason,
                            model, data, obj_body_id,
                            fields={"min_contact_dist": float(min_cdist),
                                    "contact_force_n_mean": float(fn_mean),
                                    "contact_force_n_count": int(fn_cnt)})
            write_skipped_stages(stage_w, stage_f, scene_name, obj_body_name, cand_id,
                                 idx, stage_name, reason, model, data, obj_body_id)
            log_stage("Stage 3 失败：no_contact（关闭夹爪后没有建立指尖-物体接触）")
            return final_fail_dict(stage_name, reason, has_contact=0,
                                   min_contact_dist=min_cdist,
                                   contact_force_n_mean=fn_mean, contact_force_n_count=fn_cnt)

        write_stage_row(stage_w, stage_f, scene_name, obj_body_name, cand_id,
                        idx, stage_name, True, "",
                        model, data, obj_body_id,
                        fields={"min_contact_dist": float(min_cdist),
                                "contact_force_n_mean": float(fn_mean),
                                "contact_force_n_count": int(fn_cnt)})
        log_stage("Stage 3 完成：已建立接触。")
    except Exception as e:
        reason = f"ik_fail:{type(e).__name__}"
        write_stage_row(stage_w, stage_f, scene_name, obj_body_name, cand_id,
                        idx, stage_name, False, reason, model, data, obj_body_id, fields=None)
        write_skipped_stages(stage_w, stage_f, scene_name, obj_body_name, cand_id,
                             idx, stage_name, reason, model, data, obj_body_id)
        log_stage(f"Stage 3 失败：{reason}")
        return final_fail_dict(stage_name, reason)

    # ---- Closed-chain reference ----
    mujoco.mj_forward(model, data)
    obj_pos = data.xpos[obj_body_id].copy()
    obj_rot = data.xmat[obj_body_id].copy().reshape(3,3)
    obj_se3 = to_mink_se3(obj_pos, obj_rot)

    left_se3 = ur5LeftArm.get_se3(data)
    right_se3 = ur5RightArm.get_se3(data)

    T_obj_to_left_init = obj_se3.inverse().multiply(left_se3)
    T_obj_to_right_init = obj_se3.inverse().multiply(right_se3)
    T_rel_target = T_obj_to_left_init.inverse().multiply(T_obj_to_right_init)

    task_relative = mink.RelativeFrameTask(
        frame_name=ur5RightArm.attachment_site_name, frame_type="site",
        root_name=ur5LeftArm.attachment_site_name, root_type="site",
        position_cost=50.0, orientation_cost=5.0, lm_damping=1.0
    )
    task_relative.set_target(T_rel_target)
    tasks_cc = tasks + [task_relative]

    # ---------- Stage 4: lift ----------
    idx, stage_name = STAGE_LIST[3]
    log_stage("开始 Stage 4: lift，抬升物体...")
    steps_lift = int(1.0 / rate.dt)

    try:
        for k in range(steps_lift):
            if (viewer is not None) and (not viewer_is_running(viewer)):
                if allow_headless:
                    viewer = None
                else:
                    reason = "viewer_closed"
                    write_stage_row(stage_w, stage_f, scene_name, obj_body_name, cand_id,
                                    idx, stage_name, False, reason, model, data, obj_body_id, fields=None)
                    write_skipped_stages(stage_w, stage_f, scene_name, obj_body_name, cand_id,
                                         idx, stage_name, reason, model, data, obj_body_id)
                    log_stage("Stage 4 失败：viewer_closed")
                    return final_fail_dict(stage_name, reason, has_contact=1)

            alpha = (k + 1) / steps_lift
            target_pos = obj_pos.copy()
            target_pos[2] += alpha * LIFT_HEIGHT
            T_world_obj = to_mink_se3(target_pos, obj_rot)

            target_left = T_world_obj.multiply(T_obj_to_left_init)
            target_right = T_world_obj.multiply(T_obj_to_right_init)
            tasks_cc[0].set_target(target_left)
            tasks_cc[1].set_target(target_right)

            st = step_ik_and_sim(model, data, configuration, tasks_cc, limits, solver, rate,
                                 arm_infos, GRIPPER_CLOSE, physics_steps)
            maybe_update_markers()
            viewer_safe_sync(viewer)

            if st in ("quit", "skip"):
                write_stage_row(stage_w, stage_f, scene_name, obj_body_name, cand_id,
                                idx, stage_name, False, st, model, data, obj_body_id, fields=None)
                write_skipped_stages(stage_w, stage_f, scene_name, obj_body_name, cand_id,
                                     idx, stage_name, st, model, data, obj_body_id)
                log_stage(f"Stage 4 失败：{st}")
                return final_fail_dict(stage_name, st, has_contact=1)

            mujoco.mj_forward(model, data)
            still_contact = contact_exists(data, fingertip_geom_ids, obj_geom_ids)
            z_now = float(data.xpos[obj_body_id][2])
            if not still_contact:
                reason = "dropped" if z_now < z0 + 0.02 else "lost_contact"
                write_stage_row(stage_w, stage_f, scene_name, obj_body_name, cand_id,
                                idx, stage_name, False, reason, model, data, obj_body_id,
                                fields={"lift_height": float(z_now - z0)})
                write_skipped_stages(stage_w, stage_f, scene_name, obj_body_name, cand_id,
                                     idx, stage_name, reason, model, data, obj_body_id)
                log_stage(f"Stage 4 失败：{reason}")
                return final_fail_dict(stage_name, reason, has_contact=1,
                                       lift_height=float(z_now - z0))

        mujoco.mj_forward(model, data)
        z1 = float(data.xpos[obj_body_id][2])
        lift_height = float(z1 - z0)

        # rel error snapshot
        left_now = ur5LeftArm.get_se3(data)
        right_now = ur5RightArm.get_se3(data)
        T_rel_now = left_now.inverse().multiply(right_now)
        rel_pos_err, rel_rot_err_deg = se3_rel_error(T_rel_now, T_rel_target)

        write_stage_row(stage_w, stage_f, scene_name, obj_body_name, cand_id,
                        idx, stage_name, True, "",
                        model, data, obj_body_id,
                        fields={"lift_height": lift_height,
                                "rel_pos_err": rel_pos_err,
                                "rel_rot_err_deg": rel_rot_err_deg})
        log_stage("Stage 4 完成。")
        log_kv("抬升结果", lift_height=f"{lift_height:.3f} m",
               rel_pos_err=f"{rel_pos_err:.4f}", rel_rot_err_deg=f"{rel_rot_err_deg:.2f}")

    except Exception as e:
        reason = f"ik_fail:{type(e).__name__}"
        write_stage_row(stage_w, stage_f, scene_name, obj_body_name, cand_id,
                        idx, stage_name, False, reason, model, data, obj_body_id, fields=None)
        write_skipped_stages(stage_w, stage_f, scene_name, obj_body_name, cand_id,
                             idx, stage_name, reason, model, data, obj_body_id)
        log_stage(f"Stage 4 失败：{reason}")
        return final_fail_dict(stage_name, reason, has_contact=1)

    # ---------- Stage 5: hold ----------
    idx, stage_name = STAGE_LIST[4]
    log_stage("开始 Stage 5: hold，保持并统计滑移/稳定性指标...")

    hold_steps = int(HOLD_TIME / rate.dt)
    hold_duration = 0.0
    max_slip = 0.0

    # stability stats
    obj_v_sq_sum = 0.0
    obj_w_sq_sum = 0.0
    hold_count = 0
    contact_frames = 0
    min_contact_dist = 1e9

    fn_sum = 0.0
    fn_cnt_sum = 0

    # keep object at lifted pose (ideal)
    T_world_obj_final = to_mink_se3(obj_pos + np.array([0, 0, LIFT_HEIGHT]), obj_rot)

    # rel error final snapshot
    rel_pos_err_final = rel_pos_err
    rel_rot_err_deg_final = rel_rot_err_deg

    try:
        for _ in range(hold_steps):
            if (viewer is not None) and (not viewer_is_running(viewer)):
                if allow_headless:
                    viewer = None
                else:
                    break

            target_left = T_world_obj_final.multiply(T_obj_to_left_init)
            target_right = T_world_obj_final.multiply(T_obj_to_right_init)
            tasks_cc[0].set_target(target_left)
            tasks_cc[1].set_target(target_right)

            st = step_ik_and_sim(model, data, configuration, tasks_cc, limits, solver, rate,
                                 arm_infos, GRIPPER_CLOSE, physics_steps)
            maybe_update_markers()
            viewer_safe_sync(viewer)

            if st in ("quit", "skip"):
                write_stage_row(stage_w, stage_f, scene_name, obj_body_name, cand_id,
                                idx, stage_name, False, st, model, data, obj_body_id,
                                fields={"lift_height": lift_height,
                                        "hold_duration": hold_duration,
                                        "max_slip": max_slip})
                # Stage 5 是最后一个阶段，无需 skipped
                log_stage(f"Stage 5 失败：{st}")
                return final_fail_dict(stage_name, st, has_contact=1,
                                       lift_height=lift_height,
                                       hold_duration=hold_duration,
                                       max_slip=max_slip)

            mujoco.mj_forward(model, data)

            # object velocities
            cvel6 = data.cvel[obj_body_id].copy()   # [w(3), v(3)] in body frame
            w = cvel6[:3]
            v = cvel6[3:]
            obj_v_sq_sum += float(v @ v)
            obj_w_sq_sum += float(w @ w)
            hold_count += 1

            # slip
            obj_pos_now = data.xpos[obj_body_id].copy()
            obj_rot_now = data.xmat[obj_body_id].copy().reshape(3,3)
            obj_se3_now = to_mink_se3(obj_pos_now, obj_rot_now)
            left_se3_now = ur5LeftArm.get_se3(data)
            right_se3_now = ur5RightArm.get_se3(data)
            dL, dR = calc_slip(obj_se3_now, left_se3_now, right_se3_now,
                               T_obj_to_left_init, T_obj_to_right_init)
            max_slip = max(max_slip, dL, dR)

            # rel errors
            T_rel_now = left_se3_now.inverse().multiply(right_se3_now)
            rel_pos_err_final, rel_rot_err_deg_final = se3_rel_error(T_rel_now, T_rel_target)

            # contact ratio + min dist
            still_contact = contact_exists(data, fingertip_geom_ids, obj_geom_ids)
            if still_contact:
                contact_frames += 1

            dmin_step, _ = min_contact_dist_between(data, fingertip_geom_ids, obj_geom_ids)
            min_contact_dist = min(min_contact_dist, float(dmin_step))

            fn_mean_step, fn_cnt_step = mean_contact_force_norm(model, data, fingertip_geom_ids, obj_geom_ids)
            if fn_cnt_step > 0:
                fn_sum += fn_mean_step
                fn_cnt_sum += 1

            # drop check
            z_now = float(data.xpos[obj_body_id][2])
            if (not still_contact) and (z_now < z0 + 0.02):
                log_stage("Stage 5 提前结束：dropped（保持期间掉落）")
                break

            hold_duration += rate.dt

        obj_v_rms = float(np.sqrt(obj_v_sq_sum / max(1, hold_count)))
        obj_w_rms = float(np.sqrt(obj_w_sq_sum / max(1, hold_count)))
        contact_ratio = float(contact_frames / max(1, hold_count))
        min_contact_dist = float(min_contact_dist) if min_contact_dist < 1e8 else 999.0
        contact_force_n_mean = float(fn_sum / fn_cnt_sum) if fn_cnt_sum > 0 else 0.0
        contact_force_n_count = int(fn_cnt_sum)

        # success criterion
        success = int(lift_height >= 0.10 and hold_duration > 0.0)

        write_stage_row(stage_w, stage_f, scene_name, obj_body_name, cand_id,
                        idx, stage_name, True, "",
                        model, data, obj_body_id,
                        fields={"lift_height": lift_height,
                                "hold_duration": hold_duration,
                                "max_slip": max_slip,
                                "obj_v_rms": obj_v_rms,
                                "obj_w_rms": obj_w_rms,
                                "contact_ratio": contact_ratio,
                                "min_contact_dist": min_contact_dist,
                                "rel_pos_err": rel_pos_err_final,
                                "rel_rot_err_deg": rel_rot_err_deg_final,
                                "contact_force_n_mean": contact_force_n_mean,
                                "contact_force_n_count": contact_force_n_count})
        log_stage("Stage 5 完成。")

        # Final summary
        log_stage("评估结束：最终结果 ->")
        log_kv("结果",
               success=success,
               lift_height=f"{lift_height:.3f}",
               hold=f"{hold_duration:.2f}s",
               max_slip=f"{max_slip:.4f}",
               obj_v_rms=f"{obj_v_rms:.4f}",
               obj_w_rms=f"{obj_w_rms:.4f}",
               contact_ratio=f"{contact_ratio:.2f}",
               min_contact_dist=f"{min_contact_dist:.4f}",
               rel_pos_err=f"{rel_pos_err_final:.4f}",
               rel_rot_err_deg=f"{rel_rot_err_deg_final:.2f}",
               contact_force_n_mean=f"{contact_force_n_mean:.3f}")

        return dict(
            success=success,
            has_contact=1,
            lift_height=lift_height,
            hold_duration=hold_duration,
            max_slip=max_slip,
            obj_v_rms=obj_v_rms,
            obj_w_rms=obj_w_rms,
            contact_ratio=contact_ratio,
            min_contact_dist=min_contact_dist,
            rel_pos_err=rel_pos_err_final,
            rel_rot_err_deg=rel_rot_err_deg_final,
            contact_force_n_mean=contact_force_n_mean,
            contact_force_n_count=contact_force_n_count,
            fail_stage="",
            fail_reason=""
        )

    except Exception as e:
        reason = f"ik_fail:{type(e).__name__}"
        write_stage_row(stage_w, stage_f, scene_name, obj_body_name, cand_id,
                        idx, stage_name, False, reason, model, data, obj_body_id,
                        fields={"lift_height": lift_height,
                                "hold_duration": hold_duration,
                                "max_slip": max_slip})
        log_stage(f"Stage 5 失败：{reason}")
        return final_fail_dict(stage_name, reason, has_contact=1,
                               lift_height=lift_height,
                               hold_duration=hold_duration,
                               max_slip=max_slip)


# =========================
# 主程序
# =========================
def main():
    logger = ResultLogger(out_dir="results")
    logger.save_json("run_config", {
        "scheme": "B (ground pick)",
        "NUM_CANDIDATES": NUM_CANDIDATES,
        "PREGRASP_OFFSET": PREGRASP_OFFSET,
        "LIFT_HEIGHT": LIFT_HEIGHT,
        "HOLD_TIME": HOLD_TIME,
        "CONTROL_FREQ": CONTROL_FREQ,
        "SOLVER": SOLVER,
        "MAX_VEL": float(MAX_VEL),
        "keys": {"q": "quit", "space": "pause", ".": "step", "n": "skip", "v": "markers"},
        "mujoco_version": getattr(mujoco, "__version__", "unknown"),
        "ALLOW_HEADLESS_AFTER_VIEWER_CLOSE": ALLOW_HEADLESS_AFTER_VIEWER_CLOSE,
        "CONTINUE_ON_CANDIDATE_EXCEPTION": CONTINUE_ON_CANDIDATE_EXCEPTION,
        "VERBOSE_STAGE": VERBOSE_STAGE,
        "stage_metrics_fixed_5_rows": True,
        "ENABLE_CONTACT_FORCE_STATS": ENABLE_CONTACT_FORCE_STATS
    })

    summary_all = {}

    repo_root = Path(__file__).resolve().parent
    scene_paths = sorted((repo_root / "assets/dual_arm_and_single_arm/scenes").glob("scene_*.xml"))

    for scene_path in scene_paths:
        scene_name = Path(scene_path).stem
        print(f"\n========== Scene: {scene_name} ==========")

        model = mujoco.MjModel.from_xml_path(str(Path(scene_path).resolve()))
        data = mujoco.MjData(model)
        configuration = mink.Configuration(model)

        sim_dt = model.opt.timestep
        physics_steps = int(np.ceil((1.0 / CONTROL_FREQ) / sim_dt))

        # robots
        ur5LeftArm = RobotArm('ur5_left', ur5_left_arm_joint_names, LEFT_EE_SITE,
                              ur5_left_arm_first_actuator_name, ur5_left_gripper_actuator_name)
        ur5RightArm = RobotArm('ur5_right', ur5_right_arm_joint_names, RIGHT_EE_SITE,
                               ur5_right_arm_first_actuator_name, ur5_right_gripper_actuator_name)
        ur5BaseBody = RobotArm('base', base_joint_name, None, base_actuator_name)

        ur5LeftArm.initialize(model)
        ur5RightArm.initialize(model)
        ur5BaseBody.initialize(model)

        arm_infos = [ur5BaseBody.get_control_info(),
                     ur5LeftArm.get_control_info(),
                     ur5RightArm.get_control_info()]

        all_joint_names = base_joint_name + ur5_left_arm_joint_names + ur5_right_arm_joint_names
        tasks, limits = init_tasks(configuration, model, all_joint_names,
                                   [LEFT_EE_SITE, RIGHT_EE_SITE], MAX_VEL)

        fingertip_geom_ids = get_geom_ids(model, LEFT_FINGERTIP_GEOMS + RIGHT_FINGERTIP_GEOMS)

        obj_body_id, obj_body_name, obj_geom_ids, obj_geom_names, obj_primary_geom_id = discover_freejoint_object(model)
        print("[INFO] object body:", obj_body_name, "geom_names:", obj_geom_names)

        # reset keyboard state
        with state_lock:
            ctrl_state["quit"] = False
            ctrl_state["paused"] = False
            ctrl_state["step_once"] = False
            ctrl_state["skip"] = False
            ctrl_state["show"] = False

        # Candidate-level CSV
        cand_fields = [
            "scene", "object_body", "candidate_id", "pair_dist",

            # world
            "left_x_w","left_y_w","left_z_w","right_x_w","right_y_w","right_z_w",
            # center
            "left_x_center","left_y_center","left_z_center","right_x_center","right_y_center","right_z_center",
            # object frame
            "left_x_obj","left_y_obj","left_z_obj","right_x_obj","right_y_obj","right_z_obj",
            # approach (world z-axis of grasp frame)
            "left_ax_w","left_ay_w","left_az_w","right_ax_w","right_ay_w","right_az_w",

            # metrics
            "success","has_contact","lift_height","hold_duration","max_slip",
            "obj_v_rms","obj_w_rms","contact_ratio","min_contact_dist",
            "rel_pos_err","rel_rot_err_deg",
            "contact_force_n_mean","contact_force_n_count",
            "fail_stage","fail_reason"
        ]
        csv_f, csv_w, csv_path = logger.open_csv(scene_name, cand_fields)

        # Stage-level CSV (固定 5 行/ candidate)
        stage_f, stage_w, stage_path = logger.open_csv(f"{scene_name}_stage_metrics", STAGE_FIELDNAMES)

        rows = []
        viewer = None

        try:
            with mujoco.viewer.launch_passive(
                model, data,
                show_left_ui=False, show_right_ui=False,
                key_callback=key_callback
            ) as v:
                viewer = v
                rate = RateLimiter(frequency=CONTROL_FREQ, warn=False)

                # settle
                print("正在进行物理预热，让物体自然落位...")
                settle_free_fall(model, data, steps=500)
                init_state = save_state_full(model, data)
                print("预热完成。")

                # compute AABB & object pose after settling
                mins, maxs, center, extent = get_object_aabb_world(model, data, obj_primary_geom_id)
                mujoco.mj_forward(model, data)
                obj_pos0 = data.xpos[obj_body_id].copy()
                obj_rot0 = data.xmat[obj_body_id].copy().reshape(3,3)

                candidates = build_grasp_candidates_adaptive(
                    mins, maxs, center, extent,
                    obj_pos_w=obj_pos0, obj_rot_w=obj_rot0,
                    num_candidates=NUM_CANDIDATES
                )
                print(f"[INFO] candidates: {len(candidates)}")
                print("[INFO] Keys: q quit | space pause | . step | n skip | v toggle markers")

                for i, cand in enumerate(candidates):
                    with state_lock:
                        if ctrl_state["quit"]:
                            viewer_safe_close(viewer)
                            break
                        ctrl_state["skip"] = False

                    load_state_full(model, data, init_state)

                    try:
                        metrics = evaluate_candidate(
                            cand, scene_name, obj_body_name, i,
                            model, data, configuration, tasks, limits, SOLVER,
                            viewer if viewer_is_running(viewer) else (None if ALLOW_HEADLESS_AFTER_VIEWER_CLOSE else viewer),
                            rate, physics_steps,
                            ur5LeftArm, ur5RightArm, arm_infos,
                            obj_body_id, obj_geom_ids,
                            fingertip_geom_ids,
                            stage_w=stage_w, stage_f=stage_f
                        )
                    except Exception as e:
                        tb = traceback.format_exc(limit=3)
                        print(f"[ERROR] candidate {i} exception:\n{tb}")

                        # 发生“未捕获异常”时：也要保证 stage 写满 5 行
                        # 这里我们写 Stage1 为失败，后续全部 skipped
                        reason = f"exception:{type(e).__name__}"
                        write_stage_row(stage_w, stage_f, scene_name, obj_body_name, i,
                                        1, "Stage1_pregrasp", False, reason,
                                        model, data, obj_body_id, fields=None)
                        write_skipped_stages(stage_w, stage_f, scene_name, obj_body_name, i,
                                             1, "Stage1_pregrasp", reason,
                                             model, data, obj_body_id)

                        metrics = dict(
                            success=0, has_contact=0, lift_height=0.0, hold_duration=0.0, max_slip=999.0,
                            obj_v_rms=999.0, obj_w_rms=999.0, contact_ratio=0.0, min_contact_dist=999.0,
                            rel_pos_err=999.0, rel_rot_err_deg=999.0,
                            contact_force_n_mean=0.0, contact_force_n_count=0,
                            fail_stage="exception",
                            fail_reason=f"{type(e).__name__}: {e}"
                        )
                        if not CONTINUE_ON_CANDIDATE_EXCEPTION:
                            pass

                    row = {
                        "scene": scene_name,
                        "object_body": obj_body_name,
                        "candidate_id": i,
                        "pair_dist": float(np.linalg.norm(cand.left_pos_w - cand.right_pos_w)),

                        "left_x_w": float(cand.left_pos_w[0]), "left_y_w": float(cand.left_pos_w[1]), "left_z_w": float(cand.left_pos_w[2]),
                        "right_x_w": float(cand.right_pos_w[0]), "right_y_w": float(cand.right_pos_w[1]), "right_z_w": float(cand.right_pos_w[2]),

                        "left_x_center": float(cand.left_pos_center[0]), "left_y_center": float(cand.left_pos_center[1]), "left_z_center": float(cand.left_pos_center[2]),
                        "right_x_center": float(cand.right_pos_center[0]), "right_y_center": float(cand.right_pos_center[1]), "right_z_center": float(cand.right_pos_center[2]),

                        "left_x_obj": float(cand.left_pos_obj[0]), "left_y_obj": float(cand.left_pos_obj[1]), "left_z_obj": float(cand.left_pos_obj[2]),
                        "right_x_obj": float(cand.right_pos_obj[0]), "right_y_obj": float(cand.right_pos_obj[1]), "right_z_obj": float(cand.right_pos_obj[2]),

                        "left_ax_w": float(cand.left_rot_w[0, 2]), "left_ay_w": float(cand.left_rot_w[1, 2]), "left_az_w": float(cand.left_rot_w[2, 2]),
                        "right_ax_w": float(cand.right_rot_w[0, 2]), "right_ay_w": float(cand.right_rot_w[1, 2]), "right_az_w": float(cand.right_rot_w[2, 2]),
                    }
                    row.update(metrics)

                    csv_w.writerow(row)
                    csv_f.flush()
                    stage_f.flush()
                    rows.append(row)

                    if (i + 1) % 10 == 0:
                        succ = sum(r["success"] for r in rows)
                        print(f"  progress {i+1}/{len(candidates)} | success so far: {succ}")

                    if str(metrics.get("fail_stage", "")) == "exception" and (not CONTINUE_ON_CANDIDATE_EXCEPTION):
                        break

        finally:
            csv_f.close()
            stage_f.close()

        if not rows:
            summary_all[scene_name] = {
                "csv": str(csv_path),
                "stage_csv": str(stage_path),
                "num_candidates": 0,
                "success_rate": 0.0,
                "top5": []
            }
            continue

        rows_sorted = sorted(rows, key=lambda r: (-r["success"], -r["hold_duration"], r["max_slip"]))
        top5 = rows_sorted[:5]
        success_rate = float(sum(r["success"] for r in rows) / len(rows))

        summary_all[scene_name] = {
            "csv": str(csv_path),
            "stage_csv": str(stage_path),
            "num_candidates": int(len(rows)),
            "success_rate": success_rate,
            "top5": top5
        }
        logger.save_json(f"top5_{scene_name}", top5)

        print(f"[INFO] Saved {scene_name} CSV:", csv_path)
        print(f"[INFO] Saved {scene_name} Stage CSV:", stage_path)
        print(f"[INFO] success_rate={success_rate:.3f}")

    logger.save_json("summary_all", summary_all)
    print("\n[DONE] Results saved in:", logger.root)


if __name__ == "__main__":
    main()
