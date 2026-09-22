"""Chapter 4 RGB-D front end for the existing five-stage grasp evaluator.

All geometry uses MuJoCo world coordinates. Camera images use top-left pixel origins;
MuJoCo cameras look down local -Z with local +Y pointing up.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


CAMERAS = ("grasp_top", "grasp_front", "grasp_left", "grasp_right", "grasp_back")


@dataclass(frozen=True)
class VisionConfig:
    width: int = 320
    height: int = 240
    voxel: float = 0.004
    min_depth: float = 0.02
    max_depth: float = 2.0
    floor_z: float = 0.003
    normal_radius: float = 0.025
    min_neighbors: int = 8
    gripper_min_width: float = 0.015
    gripper_max_width: float = 0.085
    finger_thickness: float = 0.012
    finger_depth: float = 0.025
    surface_offset: float = 0.006
    min_pair_distance: float = 0.08
    max_pair_distance: float = 0.5
    max_approach_dot: float = 0.25
    reach_radius: float = 1.1
    min_score: float = 0.2
    seed: int = 0


@dataclass(frozen=True)
class SingleGrasp:
    position: np.ndarray
    rotation: np.ndarray
    width: float
    score: float
    contacts: np.ndarray


def backproject(depth, mask, rotation_cw, translation_cw, fovy_degrees,
                min_depth=0.02, max_depth=2.0):
    """Metric optical depth -> world XYZ; MuJoCo camera axes are (right, up, back)."""
    h, w = depth.shape
    v, u = np.nonzero(mask & np.isfinite(depth) &
                      (depth > min_depth) & (depth < max_depth))
    if not len(u):
        return np.empty((0, 3), dtype=float)
    z = depth[v, u]
    fy = h / (2 * np.tan(np.deg2rad(fovy_degrees) / 2))
    fx = fy  # square pixels; vertical field of view defines the focal length
    camera_xyz = np.stack(((u + 0.5 - w / 2) * z / fx,
                           -(v + 0.5 - h / 2) * z / fy, -z), axis=-1)
    return camera_xyz @ np.asarray(rotation_cw).T + translation_cw


def voxel_downsample(points, voxel):
    if not len(points):
        return np.empty((0, 3), dtype=float)
    keys, inverse = np.unique(np.floor(points / voxel).astype(np.int64),
                              axis=0, return_inverse=True)
    sums = np.zeros((len(keys), 3))
    np.add.at(sums, inverse, points)
    return sums / np.bincount(inverse)[:, None]


def capture_object_cloud(model, data, geom_ids, config=VisionConfig(), cameras=CAMERAS):
    """Render five RGB-D views and use geom segmentation as the object mask.

    Returns an object-only cloud and a per-camera bundle of RGB, metric depth, mask,
    intrinsics and camera-to-world pose. Rendering requires an OpenGL context.
    """
    import mujoco

    renderer = mujoco.Renderer(model, height=config.height, width=config.width)
    views, clouds = {}, []
    try:
        for name in cameras:
            camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            if camera_id < 0:
                raise ValueError(f"Camera {name!r} is absent from scene XML")
            renderer.disable_depth_rendering()
            renderer.disable_segmentation_rendering()
            renderer.update_scene(data, camera=name)
            rgb = renderer.render().copy()
            renderer.enable_depth_rendering()
            renderer.update_scene(data, camera=name)
            depth = renderer.render().copy()
            renderer.disable_depth_rendering()
            renderer.enable_segmentation_rendering()
            renderer.update_scene(data, camera=name)
            segmentation = renderer.render().copy()
            renderer.disable_segmentation_rendering()
            # Segmentation channel 0 is object id, channel 1 is object type.
            mask = (np.isin(segmentation[..., 0], list(geom_ids)) &
                    (segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)))
            rotation = data.cam_xmat[camera_id].reshape(3, 3).copy()
            translation = data.cam_xpos[camera_id].copy()
            fovy = float(model.cam_fovy[camera_id])
            fy = config.height / (2 * np.tan(np.deg2rad(fovy) / 2))
            intrinsics = np.array([[fy, 0, config.width / 2],
                                   [0, fy, config.height / 2], [0, 0, 1]])
            cloud = backproject(depth, mask, rotation, translation, fovy,
                                config.min_depth, config.max_depth)
            views[name] = dict(rgb=rgb, depth=depth, mask=mask,
                               K=intrinsics, R_cw=rotation, t_cw=translation)
            clouds.append(cloud)
    finally:
        renderer.close()
    points = voxel_downsample(np.concatenate(clouds), config.voxel)
    points = points[np.isfinite(points).all(axis=1) & (points[:, 2] > config.floor_z)]
    return points, views


def estimate_normals(points, center, radius=0.025, min_neighbors=8):
    """Local PCA with normals oriented away from the object cloud centroid."""
    tree = cKDTree(points)
    normals = np.zeros_like(points)
    for i, neighbors in enumerate(tree.query_ball_point(points, radius)):
        if len(neighbors) < min_neighbors:
            continue
        patch = points[neighbors]
        centered = patch - patch.mean(axis=0)
        _, vectors = np.linalg.eigh(centered.T @ centered / len(patch))
        normal = vectors[:, 0]
        if normal @ (points[i] - center) < 0:
            normal = -normal
        normals[i] = normal
    return normals


def filter_cloud(points, radius=0.02, neighbors=6, std_ratio=2.0):
    """Statistical and radius outlier removal after object-ID and floor masking."""
    if len(points) <= neighbors:
        return points
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=neighbors + 1)
    mean_dist = distances[:, 1:].mean(axis=1)
    keep = mean_dist <= mean_dist.mean() + std_ratio * mean_dist.std()
    points = points[keep]
    if len(points) <= neighbors:
        return points
    tree = cKDTree(points)
    counts = np.array([len(ids) for ids in tree.query_ball_point(points, radius)])
    return points[counts >= neighbors]


def direction_families(points, normals, patch_size=0.025, min_points=8):
    """Voxel surface patches with average inward approach directions (report 4.6)."""
    keys, inverse = np.unique(np.floor(points / patch_size).astype(int),
                              axis=0, return_inverse=True)
    patches = []
    for i in range(len(keys)):
        ids = np.flatnonzero(inverse == i)
        if len(ids) < min_points:
            continue
        direction = -normals[ids].sum(axis=0)
        length = np.linalg.norm(direction)
        if length > 1e-8:
            patches.append((points[ids].mean(axis=0), direction / length))
    return np.asarray(patches).reshape(-1, 2, 3)


def sample_single_grasps(points, normals, config=VisionConfig(), max_seeds=500):
    """Sample local jaw frames; accept pairs of opposed, supported contact patches."""
    rng = np.random.default_rng(config.seed)
    tree = cKDTree(points)
    order = rng.permutation(len(points))[:max_seeds]
    grasps = []
    for index in order:
        outward = normals[index]
        if np.linalg.norm(outward) < 0.9 or outward[2] < -0.4:
            continue
        x = -outward  # contact-to-contact direction is the jaw closing axis
        tangent = np.array([0., 0., 1.])
        if abs(tangent @ x) > 0.9:
            tangent = np.array([1., 0., 0.])
        tangent -= (tangent @ x) * x
        tangent /= np.linalg.norm(tangent)
        for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            approach = np.cos(angle) * tangent + np.sin(angle) * np.cross(x, tangent)
            if approach[2] > 0.25:
                continue
            y = np.cross(approach, x)
            # Find the opposite surface close to the same approach/height slice.
            local = points - points[index]
            depth = local @ approach
            side = local @ y
            opening = local @ x
            opposite = ((depth > -config.finger_thickness) &
                        (depth < config.finger_depth) &
                        (abs(side) < config.finger_thickness) &
                        (opening > config.gripper_min_width) &
                        (opening < config.gripper_max_width) &
                        ((normals @ outward) < -0.5))
            ids = np.flatnonzero(opposite)
            if len(ids):
                other = ids[np.argmin(abs(depth[ids]))]
                width = float(opening[other])
                midpoint = (points[index] + points[other]) / 2
                contacts = np.stack((points[index], points[other]))
                support_l = len(tree.query_ball_point(points[index], config.finger_thickness))
                support_r = len(tree.query_ball_point(points[other], config.finger_thickness))
                alignment = max(0., -normals[index] @ normals[other])
                balance = max(0., 1 - abs(depth[other]) / config.finger_depth)
                support = min(1., min(support_l, support_r) / config.min_neighbors)
                score = float(0.4 * alignment + 0.3 * balance + 0.3 * support)
            else:
                # A partial RGB-D cloud often hides the opposite face. Hypothesize
                # a 4 cm jaw span on a locally supported patch; score it lower.
                width = min(0.04, config.gripper_max_width)
                local_ids = tree.query_ball_point(points[index], width)
                local = points[local_ids] - points[index]
                proj = local @ y
                plus = np.count_nonzero((proj > width / 4) & (proj < width / 2))
                minus = np.count_nonzero((proj < -width / 4) & (proj > -width / 2))
                if min(plus, minus) < config.min_neighbors:
                    continue
                midpoint = points[index]
                contacts = np.stack((midpoint - y * width / 2,
                                     midpoint + y * width / 2))
                score = float(0.2 + 0.25 * min(1., min(plus, minus) / 20))
            if score < config.min_score:
                continue
            if len(ids):
                rotation = np.column_stack((x, y, approach))
            else:
                # For patch support the jaw axis lies in the surface tangent plane.
                patch_x, patch_z = y, -outward
                rotation = np.column_stack((patch_x, np.cross(patch_z, patch_x), patch_z))
            grasps.append(SingleGrasp(midpoint - config.surface_offset * rotation[:, 2],
                                      rotation, width, score, contacts))
    return sorted(grasps, key=lambda g: -g.score)


def pair_grasps(grasps, obj_position, obj_rotation, center,
                left_reference, right_reference, config=VisionConfig(), limit=100):
    """Pair two distinct local grasps and adapt them to the original evaluator."""
    from batch_grasp_pair_eval_schemeB import GraspCandidate

    output = []
    # Spatial deduplication prevents 100 copies of one surface patch.
    seen = set()
    for i, a in enumerate(grasps[:400]):
        for b in grasps[i + 1:400]:
            left, right = (a, b) if a.position[1] <= b.position[1] else (b, a)
            delta = np.linalg.norm(left.position - right.position)
            if not config.min_pair_distance <= delta <= config.max_pair_distance:
                continue
            if left.rotation[:, 2] @ right.rotation[:, 2] > config.max_approach_dot:
                continue
            if (np.linalg.norm(left.position - left_reference) > config.reach_radius or
                    np.linalg.norm(right.position - right_reference) > config.reach_radius):
                continue
            key = (tuple(np.round(left.position / 0.02).astype(int)),
                   tuple(np.round(right.position / 0.02).astype(int)))
            if key in seen:
                continue
            seen.add(key)
            output.append(GraspCandidate(
                left_pos_w=left.position, right_pos_w=right.position,
                left_rot_w=left.rotation, right_rot_w=right.rotation,
                left_pos_center=left.position - center,
                right_pos_center=right.position - center,
                left_pos_obj=obj_rotation.T @ (left.position - obj_position),
                right_pos_obj=obj_rotation.T @ (right.position - obj_position),
                left_rot_obj=obj_rotation.T @ left.rotation,
                right_rot_obj=obj_rotation.T @ right.rotation,
                left_width=left.width, right_width=right.width,
                pair_score=(left.score + right.score) / 2))
            if len(output) >= limit:
                return output
    return output


def visual_candidates(model, data, geom_ids, obj_position, obj_rotation,
                      left_reference, right_reference, config=VisionConfig(), limit=100):
    points, views = capture_object_cloud(model, data, geom_ids, config)
    points = filter_cloud(points)
    if len(points) < config.min_neighbors:
        raise ValueError(f"Only {len(points)} segmented object points; check cameras/GL")
    center = points.mean(axis=0)
    normals = estimate_normals(points, center, config.normal_radius, config.min_neighbors)
    families = direction_families(points, normals)
    singles = sample_single_grasps(points, normals, config)
    pairs = pair_grasps(singles, obj_position, obj_rotation, center,
                        left_reference, right_reference, config, limit)
    return pairs, dict(points=points, normals=normals, families=families, views=views,
                       single_count=len(singles))
