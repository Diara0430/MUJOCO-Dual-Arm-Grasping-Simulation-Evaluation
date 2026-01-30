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
        print("Usage: python inspect_black_artifacts.py /path/to/scene.xml")
        sys.exit(1)

    scene = Path(sys.argv[1]).resolve()
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print("[INFO] ngeom:", model.ngeom, "nlight:", model.nlight)

    # 1) 黑色可见 geom（RGB都很低但 alpha>0）
    print("\n========== BLACKISH VISIBLE GEOMS ==========")
    cnt = 0
    for gid in range(model.ngeom):
        rgba = model.geom_rgba[gid]
        if rgba[3] > 0.05 and rgba[0] < 0.15 and rgba[1] < 0.15 and rgba[2] < 0.15:
            gname = safe_name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
            b = model.geom_bodyid[gid]
            bname = safe_name(model, mujoco.mjtObj.mjOBJ_BODY, b)
            print(f"- geom[{gid}] {gname} body={bname} rgba={rgba} group={int(model.geom_group[gid])}"
                  f" contype={int(model.geom_contype[gid])} conaff={int(model.geom_conaffinity[gid])}")
            cnt += 1
            if cnt >= 50:
                print("... (truncated)")
                break
    if cnt == 0:
        print("No blackish geoms found (by heuristic).")

    # 2) 灯光阴影开关（如果 castshadow 开，会产生明显阴影）[4](https://deepwiki.com/stillonearth/MuJoCo-WASM/10-rendering)[5](https://github-wiki-see.page/m/AIMotionLab-SZTAKI/AIMotionLab-Virtual/wiki/Generating-MuJoCo-XML-models)
    print("\n========== LIGHTS (castshadow) ==========")
    for lid in range(model.nlight):
        lname = safe_name(model, mujoco.mjtObj.mjOBJ_LIGHT, lid)
        # light_castshadow 在不同版本字段略有差异；Python binding通常是 model.light_castshadow
        castshadow = None
        if hasattr(model, "light_castshadow"):
            castshadow = int(model.light_castshadow[lid])
        print(f"- light[{lid}] {lname} castshadow={castshadow}")

    print("\n[HINT]")
    print("If black artifacts are from collision meshes: set their rgba alpha=0 or contype/conaffinity group to hide.")
    print("If black artifacts are real shadows: set light castshadow=0 in your world xml to test.")

if __name__ == "__main__":
    main()