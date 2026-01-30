#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
在 tools 目录里也能直接运行的 generate_scenes.py

功能：
  - 遍历 <repo>/assets/dual_arm_and_single_arm/objects/object_sim/*/object.xml
  - 生成最简 MuJoCo scene_<obj>.xml 到 <repo>/assets/dual_arm_and_single_arm/scenes/

生成内容示例：
<mujoco model="scene_alarmclock_ur5">
  <include file="../world_pure.xml"/>
  <include file="../dual_ur5_scene.xml"/>
  <include file="../objects/object_sim/alarmclock/object.xml"/>
</mujoco>

运行方式（你要的）：
  cd /home/qianny/Desktop/dual_arm_example/tools
  python generate_scenes.py

覆盖已有文件：
  python generate_scenes.py --overwrite
"""

from pathlib import Path
import argparse
import os


def make_scene_xml(model_name: str, world_inc: str, dual_ur5_inc: str, obj_inc: str) -> str:
    """生成最简 scene.xml 字符串"""
    return f"""<mujoco model="{model_name}">

  <!-- 世界 -->
  <include file="{world_inc}"/>

  <!-- 双 UR5（不含任何 Panda 相关 include） -->
  <include file="{dual_ur5_inc}"/>

  <!-- 对象（assets + body） -->
  <include file="{obj_inc}"/>

</mujoco>
"""


def relpath(from_dir: Path, to_file: Path) -> str:
    """计算从 from_dir 到 to_file 的相对路径（用于 include file="..."）"""
    return os.path.relpath(str(to_file), start=str(from_dir))


def find_base_dir(script_dir: Path) -> Path:
    """
    自动向上查找 assets/dual_arm_and_single_arm
    这样即使你从 tools 目录运行，也能找到正确路径。
    """
    cur = script_dir
    for _ in range(8):  # 向上找最多 8 层，足够稳
        candidate = cur / "assets" / "dual_arm_and_single_arm"
        if candidate.exists():
            return candidate
        cur = cur.parent

    raise FileNotFoundError(
        "Cannot find 'assets/dual_arm_and_single_arm' by searching upward from:\n"
        f"  {script_dir}\n"
        "Please check your repo structure or pass --base-dir explicitly."
    )


def generate_scenes(object_sim_dir: Path, scenes_dir: Path, base_dir: Path, overwrite: bool = False):
    """
    遍历 object_sim 子文件夹，找到每个子文件夹中的 object.xml，
    生成 scene_<obj>.xml 到 scenes_dir
    返回：scenes_dir, generated, skipped
    """
    if not object_sim_dir.exists():
        raise FileNotFoundError(f"object_sim_dir not found: {object_sim_dir}")

    # ✅ 按你的要求：不创建任何新文件夹，scenes_dir 必须存在
    if not scenes_dir.exists():
        raise FileNotFoundError(
            f"scenes_dir does not exist (won't create new folders): {scenes_dir}\n"
            "Please make sure <repo>/assets/dual_arm_and_single_arm/scenes exists."
        )

    world_xml = base_dir / "world_pure.xml"
    dual_ur5_scene_xml = base_dir / "dual_ur5_scene.xml"

    if not world_xml.exists():
        raise FileNotFoundError(f"world_pure.xml not found: {world_xml}")
    if not dual_ur5_scene_xml.exists():
        raise FileNotFoundError(f"dual_ur5_scene.xml not found: {dual_ur5_scene_xml}")

    generated = 0
    skipped = 0

    # 只遍历第一层子目录：object_sim/<obj_name>/
    for obj_dir in sorted(object_sim_dir.iterdir()):
        if not obj_dir.is_dir():
            continue

        obj_xml = obj_dir / "object.xml"
        if not obj_xml.exists():
            skipped += 1
            continue

        obj_name = obj_dir.name
        scene_file = scenes_dir / f"scene_{obj_name}.xml"

        if scene_file.exists() and not overwrite:
            skipped += 1
            continue

        # include 路径用相对路径计算（保证从 tools 跑也正确）
        world_inc = relpath(scenes_dir, world_xml)
        dual_ur5_inc = relpath(scenes_dir, dual_ur5_scene_xml)
        obj_inc = relpath(scenes_dir, obj_xml)

        model_name = f"scene_{obj_name}_ur5"
        xml_text = make_scene_xml(model_name, world_inc, dual_ur5_inc, obj_inc)

        scene_file.write_text(xml_text, encoding="utf-8")
        generated += 1

    return scenes_dir, generated, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Generate minimal MuJoCo scene xml into existing scenes/ folder (run anywhere)"
    )
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的 scene_*.xml")
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="手动指定 base_dir（即 dual_arm_and_single_arm 目录）。默认自动向上查找 assets/dual_arm_and_single_arm",
    )
    # ✅ 保留可选：手动指定 object_sim_dir（不影响“不创建文件夹”的要求）
    parser.add_argument(
        "--object-sim-dir",
        type=str,
        default=None,
        help="手动指定 object_sim 目录（默认 base_dir/objects/object_sim）",
    )

    args = parser.parse_args()

    # 核心：永远从脚本自身位置开始推导，不依赖你当前在哪个目录运行
    script_dir = Path(__file__).resolve().parent

    base_dir = Path(args.base_dir).expanduser().resolve() if args.base_dir else find_base_dir(script_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"base_dir not found: {base_dir}")

    default_object_sim_dir = base_dir / "objects" / "object_sim"
    scenes_dir = base_dir / "scenes"  # ✅ 固定写入这里（不再支持 out-dir）

    object_sim_dir = Path(args.object_sim_dir).expanduser().resolve() if args.object_sim_dir else default_object_sim_dir

    scenes_dir, generated, skipped = generate_scenes(
        object_sim_dir=object_sim_dir,
        scenes_dir=scenes_dir,
        base_dir=base_dir,
        overwrite=args.overwrite,
    )

    print(f"[OK] base_dir        = {base_dir}")
    print(f"[OK] object_sim_dir  = {object_sim_dir}")
    print(f"[OK] scenes_dir      = {scenes_dir} (existing only)")
    print(f"[STAT] generated={generated}, skipped={skipped}")


if __name__ == "__main__":
    main()
