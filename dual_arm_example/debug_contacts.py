
import mujoco
import numpy as np
from pathlib import Path
import sys
import traceback

def gname(model, gid):
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)

def list_named_geoms(model):
    names = []
    for i in range(model.ngeom):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if n:
            names.append(n)
    return sorted(set(names))

def find_object_geom_id(model, prefer="object_geom"):
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, prefer)
    if gid != -1:
        return gid, prefer

    # fallback：挑名字里带 box/object/big/small 的
    candidates = []
    for i in range(model.ngeom):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if not n:
            continue
        key = n.lower()
        if ("box" in key) or ("object" in key) or ("big" in key) or ("small" in key):
            candidates.append(n)
    candidates = sorted(set(candidates))
    if candidates:
        name = candidates[0]
        gid2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        return gid2, name
    return -1, None

def find_body_of_geom(model, geom_id):
    # geom_id -> geom_bodyid
    bid = model.geom_bodyid[geom_id]
    bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
    return bid, bname

def main():
    repo_root = Path(__file__).resolve().parent
    default_scene = repo_root / "assets" / "dual_arm_and_single_arm" / "scenes" / "scene_big_box.xml"

    scene_xml = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else default_scene
    if not scene_xml.is_absolute():
        scene_xml = (repo_root / scene_xml).resolve()

    prefer_geom = sys.argv[2] if len(sys.argv) > 2 else "object_geom"
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 200

    print("[INFO] repo_root:", repo_root)
    print("[INFO] scene_xml :", scene_xml)
    print("[INFO] prefer_geom:", prefer_geom)
    print("[INFO] steps:", steps)

    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    print("gravity:", model.opt.gravity, "timestep:", model.opt.timestep)

    obj_geom_id, obj_geom_name = find_object_geom_id(model, prefer=prefer_geom)
    if obj_geom_id == -1:
        print("[ERROR] Cannot find object geom. Named geoms are:")
        for n in list_named_geoms(model)[:200]:
            print("  -", n)
        raise ValueError("No suitable object geom found. Pass geom name as argv[2].")

    obj_body_id, obj_body_name = find_body_of_geom(model, obj_geom_id)
    print(f"[INFO] Using object geom: {obj_geom_name} (geom_id={obj_geom_id})")
    print(f"[INFO] Object body: {obj_body_name} (body_id={obj_body_id})")

    # 清控制，避免机器人影响
    data.ctrl[:] = 0
    if hasattr(data, "act") and data.act is not None:
        data.act[:] = 0
    data.qfrc_applied[:] = 0
    mujoco.mj_forward(model, data)

    for step in range(steps):
        mujoco.mj_step(model, data)

        z = data.xpos[obj_body_id][2]
        vz = data.cvel[obj_body_id][2]  # 注意：cvel是6D速度，索引2是线速度z（一般够用）
        print(f"step={step:4d} z={z:.4f} vz={vz:.4f} ncon={data.ncon}")

        # 一旦出现 contact，打印所有与该 geom 有关的 contact
        if data.ncon > 0:
            for i in range(data.ncon):
                c = data.contact[i]
                if c.geom1 == obj_geom_id or c.geom2 == obj_geom_id:
                    n1, n2 = gname(model, c.geom1), gname(model, c.geom2)
                    print(f"  contact: {n1} <-> {n2}, dist={c.dist:.6f}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[ERROR] Exception:", e)
        traceback.print_exc()
        sys.exit(1)
