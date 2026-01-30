#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np
import mujoco

def safe_name(model, objtype, idx):
    try:
        return mujoco.mj_id2name(model, objtype, idx) or f"<unnamed:{idx}>"
    except Exception:
        return f"<id:{idx}>"

def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_model_visibility.py /path/to/scene.xml [name_keyword]")
        sys.exit(1)

    scene = Path(sys.argv[1]).resolve()
    kw = sys.argv[2] if len(sys.argv) >= 3 else "alarmclock"  # 默认按 alarmclock 关键字筛选

    print("[INFO] loading:", scene)
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print("[INFO] mujoco version:", getattr(mujoco, "__version__", "unknown"))
    print("[INFO] nbody:", model.nbody, "ngeom:", model.ngeom, "nmesh:", model.nmesh, "nlight:", model.nlight)

    # 1) 找到所有名字包含 kw 的 geom
    hits = []
    for gid in range(model.ngeom):
        name = safe_name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if kw.lower() in name.lower():
            hits.append(gid)

    print(f"\n========== GEOMS CONTAIN '{kw}' ({len(hits)}) ==========")
    if not hits:
        print("No geom matched. Try another keyword, e.g. 'contact0' or list all object geoms.")
    for gid in hits[:200]:
        gname = safe_name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        b = model.geom_bodyid[gid]
        bname = safe_name(model, mujoco.mjtObj.mjOBJ_BODY, b)
        gtype = int(model.geom_type[gid])
        contype = int(model.geom_contype[gid])
        conaff = int(model.geom_conaffinity[gid])
        condim = int(model.geom_condim[gid])
        group = int(model.geom_group[gid])
        rgba = model.geom_rgba[gid].copy()
        meshid = int(model.geom_dataid[gid]) if gtype == mujoco.mjtGeom.mjGEOM_MESH else -1

        gxpos = data.geom_xpos[gid].copy()
        # xmat is 9 numbers
        gxmat = data.geom_xmat[gid].copy().reshape(3,3)

        print(f"- geom[{gid}] {gname}")
        print(f"  body={bname} type={gtype} meshid={meshid}")
        print(f"  contype={contype} conaff={conaff} condim={condim} group={group}")
        print(f"  rgba={rgba}  world_pos={gxpos}")

    # 2) 检查是否存在“可视化几何”：contype=0 & conaff=0 通常是 visual-only
    visual_only = 0
    collision_only = 0
    blackish = 0
    for gid in range(model.ngeom):
        rgba = model.geom_rgba[gid]
        if model.geom_contype[gid] == 0 and model.geom_conaffinity[gid] == 0:
            visual_only += 1
        if model.geom_contype[gid] != 0 or model.geom_conaffinity[gid] != 0:
            collision_only += 1
        # 统计“很黑”的 geom（RGB都很低且alpha>0）
        if rgba[3] > 0.05 and rgba[0] < 0.15 and rgba[1] < 0.15 and rgba[2] < 0.15:
            blackish += 1

    print("\n========== GLOBAL STATS ==========")
    print("visual_only geoms (contype=0 & conaff=0):", visual_only)
    print("collision-participating geoms (contype!=0 or conaff!=0):", collision_only)
    print("blackish visible geoms (alpha>0, rgb<0.15):", blackish)

    # 3) 打印一下所有包含 kw 的 body 的位置，帮助你找“闹钟到底在哪”
    body_hits = set(model.geom_bodyid[gid] for gid in hits)
    print(f"\n========== BODIES OF '{kw}' GEOMS ({len(body_hits)}) ==========")
    for bid in sorted(list(body_hits))[:50]:
        bname = safe_name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        print(f"- body[{bid}] {bname}: xpos={data.xpos[bid]}")

if __name__ == "__main__":
    main()