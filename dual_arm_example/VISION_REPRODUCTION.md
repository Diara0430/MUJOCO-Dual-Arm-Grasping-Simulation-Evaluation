# Visual-input grasping reproduction (report Chapters 4–5)

This adds a geometric RGB-D front end to the existing five-stage physics evaluation.
The report specifies the pipeline and equations, but not camera poses, numerical
thresholds, or the exact score function. Those choices are explicit in
`visual_grasp_candidates.py` as `VisionConfig`; results are therefore a reproducible
implementation of the method, **not** a claim to reproduce the published percentages.

## Run

From `dual_arm_example/`:

```bash
python -m pip install -r requirements-vision.txt
MUJOCO_GL=egl python batch_grasp_pair_eval_schemeB.py --candidate-source vision --headless --max-candidates 10
```

On a desktop with an OpenGL display, omit `MUJOCO_GL=egl` and `--headless` to
watch the viewer. The previous AABB route remains the default. The repository
currently ships one scene, `scene_obj_big_box.xml`; other objects require their
own scene XMLs with the five named cameras (or camera names supplied to the
vision module). Add the same camera elements around each object's settled pose.

## What is implemented

1. Render RGB, metric depth, and geom segmentation from five MuJoCo cameras;
   mask the discovered object's geoms before reconstructing the world cloud.
2. Back-project using vertical FOV and MuJoCo's camera frame, voxel downsample,
   remove floor and statistical/radius outliers, estimate outward PCA normals,
   and group surface patches into inward direction families.
3. Sample oriented parallel-jaw hypotheses around local surface frames. Prefer
   observed opposing normals within the 85 mm gripper opening. When the opposite
   face is occluded, use lower-scored local patch support as a hypothesis.
4. Pair candidates by separation, approach compatibility, score, approximate
   distance-to-end-effector reach, and spatial deduplication. Convert them to
   the original evaluator's world, center-relative, and object frames.
5. Run the original pregrasp, approach, close, lift, and hold stages. Outputs in
   `results/<timestamp>/` include the original candidate/stage CSVs and summary,
   plus `<scene>_vision.npz` with RGB, depth, masks, intrinsics, camera poses,
   fused object cloud, normals, and direction families.

## Limits and adjustments

- A two-dimensional image cannot reveal an occluded contact face. Patch-supported
  candidates are geometric proposals, not proven force-closure grasps. Physics
  execution provides the actual success label.
- MuJoCo segmentation provides exact object IDs in simulation; it is an
  assumption unavailable on a real RGB-D sensor. The report also mentions an
  optional Open3D TSDF path, which is not needed for the direct voxel fusion here.
- The one supplied object is 160 × 320 × 40 mm, wider than a Robotiq 85 mm
  gripper along two axes. Lifting failure is possible even when contact occurs.
- Reachability here is the report's radial prefilter, not an IK or collision
  feasibility proof. Tune `VisionConfig` after checking contact and failure CSVs.
- Rendering needs an OpenGL context; set `MUJOCO_GL=egl` on a Linux machine
  with EGL, or use your desktop display.
