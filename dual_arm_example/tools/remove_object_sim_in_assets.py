#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


def fix_file_path(path_str: str) -> str:
    """
    删除路径中的 '/object_sim' 或 '\\object_sim'
    - 只删除这一段，不改变其它部分
    """
    if path_str is None:
        return path_str
    # 同时兼容 linux 路径和 windows 路径
    s = path_str.replace("\\object_sim", "").replace("/object_sim", "")
    return s


def patch_assets_xml(xml_path: Path) -> tuple[bool, int]:
    """
    修改一个 assets.xml：
    - 遍历 <asset> 里的所有 <mesh> 标签
    - 对 mesh 的 file 属性做 '/object_sim' 删除
    返回：(是否修改, 修改了多少处 file)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    changed = False
    n_edits = 0

    # 找 <asset> ... </asset>
    for asset in root.iter("asset"):
        # asset 下的 mesh
        for mesh in asset.iter("mesh"):
            f = mesh.get("file")
            if not f:
                continue
            new_f = fix_file_path(f)
            if new_f != f:
                mesh.set("file", new_f)
                changed = True
                n_edits += 1

    return changed, n_edits, tree


def main():
    ap = argparse.ArgumentParser(
        description="Remove '/object_sim' segment from mesh file paths in object_sim/*/assets.xml"
    )
    ap.add_argument(
        "object_sim_dir",
        type=str,
        help="object_sim 目录路径，例如: ./assets/dual_arm_and_single_arm/objects/object_sim"
    )
    ap.add_argument("--dry-run", action="store_true", help="只打印将修改哪些文件，不写回")
    ap.add_argument("--no-backup", action="store_true", help="不生成 .bak 备份（默认会备份）")
    args = ap.parse_args()

    object_sim_dir = Path(args.object_sim_dir).resolve()
    if not object_sim_dir.exists():
        raise SystemExit(f"[ERROR] object_sim_dir not found: {object_sim_dir}")

    targets = sorted(object_sim_dir.glob("*/assets.xml"))
    print(f"[INFO] object_sim_dir: {object_sim_dir}")
    print(f"[INFO] found {len(targets)} targets (pattern: */assets.xml)")

    modified_files = 0
    total_edits = 0

    for xml_path in targets:
        try:
            changed, n_edits, tree = patch_assets_xml(xml_path)
        except Exception as e:
            print(f"[SKIP] parse failed: {xml_path} | {type(e).__name__}: {e}")
            continue

        if not changed:
            continue

        print(f"[PATCH] {xml_path} | edits={n_edits}")
        total_edits += n_edits

        if args.dry_run:
            continue

        # 备份
        if not args.no_backup:
            bak = xml_path.with_suffix(xml_path.suffix + ".bak")
            if not bak.exists():
                shutil.copy2(xml_path, bak)

        # 美化缩进（Python 3.9+）
        try:
            ET.indent(tree, space="  ")
        except Exception:
            pass

        tree.write(xml_path, encoding="utf-8", xml_declaration=True)
        modified_files += 1

    print("\n========== SUMMARY ==========")
    print(f"modified_files: {modified_files} (dry-run 写回为0)")
    print(f"total_edits: {total_edits}")
    print(f"dry_run: {args.dry_run}")


if __name__ == "__main__":
    main()

#   how to use:
#        ✅ 使用方法
#    假设你的目录是：
#    ./assets/dual_arm_and_single_arm/objects/object_sim/
#    apple/assets.xml
#    alarmclock/assets.xml
#    ...

#    1）先 dry-run（只看会改哪些文件）
#    python remove_object_sim_in_assets.py \
#    ./assets/dual_arm_and_single_arm/objects/object_sim \
#     --dry-run
#    2）确认输出没问题后，正式写回（会生成 .bak 备份）
#    python remove_object_sim_in_assets.py \
#    ./assets/dual_arm_and_single_arm/objects/object_sim
#    3）不想生成备份
#    python remove_object_sim_in_assets.py \
#  ./assets/dual_arm_and_single_arm/objects/object_sim \
#  --no-backup
#    ✅ 快速验证是否生效
#    看还有没有 object_sim 段
#    grep -R --line-number "/object_sim" ./assets/dual_arm_and_single_arm/objects/object_sim/*/assets.xml
#    看改动后的路径是否正确-->
#   grep -R --line-number 'mesh .*file=' ./assets/dual_arm_and_single_arm/objects/object_sim/*/assets.xml | head#