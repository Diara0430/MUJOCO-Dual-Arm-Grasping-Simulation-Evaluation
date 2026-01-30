#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


def parse_pos3(pos_str: str):
    """解析 'x y z' -> [x,y,z] float；失败返回 None"""
    try:
        parts = pos_str.strip().split()
        if len(parts) != 3:
            return None
        return [float(parts[0]), float(parts[1]), float(parts[2])]
    except Exception:
        return None


def format_pos3(xyz):
    """[x,y,z] -> 'x y z' （尽量保持简洁）"""
    return f"{xyz[0]:g} {xyz[1]:g} {xyz[2]:g}"


def body_has_freejoint(body_el, fj_name="object_free"):
    """检查 body 下是否已有 <freejoint name='object_free'/>"""
    for ch in list(body_el):
        if ch.tag == "freejoint" and ch.get("name") == fj_name:
            return True
    return False


def insert_freejoint_after_include(body_el, include_file="body.xml", fj_name="object_free"):
    """
    在 <include file="body.xml"/> 后插入 <freejoint name="object_free"/>
    - 若已存在同名 freejoint 则不插入，返回 False
    - 若没找到 include，则不插入，返回 False
    - 插入成功返回 True
    """
    if body_has_freejoint(body_el, fj_name):
        return False

    children = list(body_el)
    for idx, ch in enumerate(children):
        if ch.tag == "include" and ch.get("file") == include_file:
            fj = ET.Element("freejoint", {"name": fj_name})
            body_el.insert(idx + 1, fj)
            return True
    return False


def patch_one_object_xml(
    xml_path: Path,
    y_new: float,
    *,
    object_body_name="object",
    include_file_name="body.xml",
    freejoint_name="object_free",
    add_pos_if_missing=False,
):
    """
    修改一个 object.xml：
    - body name="object" 的 pos.y 改成 y_new
    - 若没有 <freejoint name="object_free"/>，则在 <include file="body.xml"/> 后插入
    返回：(changed_pos, inserted_freejoint, warnings)
    """
    warnings = []
    tree = ET.parse(xml_path)
    root = tree.getroot()

    changed_pos = False
    inserted_fj = False
    found_object_body = False

    for body in root.iter("body"):
        if body.get("name") != object_body_name:
            continue

        found_object_body = True

        # 1) 修改 pos.y
        pos_str = body.get("pos")
        if pos_str is None:
            if add_pos_if_missing:
                # 没有 pos 就补一个：x=0, y=y_new, z=0.05（你可按需改默认 z）
                body.set("pos", format_pos3([0.0, y_new, 0.05]))
                changed_pos = True
            else:
                warnings.append(f"{xml_path.name}: body '{object_body_name}' has no pos=..., skipped pos edit")
        else:
            xyz = parse_pos3(pos_str)
            if xyz is None:
                warnings.append(f"{xml_path.name}: pos='{pos_str}' parse failed, skipped pos edit")
            else:
                if xyz[1] != y_new:
                    xyz[1] = y_new
                    body.set("pos", format_pos3(xyz))
                    changed_pos = True

        # 2) 插入 freejoint（若不存在）
        if not body_has_freejoint(body, freejoint_name):
            ok = insert_freejoint_after_include(
                body,
                include_file=include_file_name,
                fj_name=freejoint_name
            )
            if ok:
                inserted_fj = True
            else:
                # 没有 include file="body.xml" 就无法按你的要求“下一行插入”
                warnings.append(
                    f"{xml_path.name}: <include file='{include_file_name}'/> not found under body '{object_body_name}', "
                    f"freejoint NOT inserted"
                )

    if not found_object_body:
        warnings.append(f"{xml_path.name}: no <body name='{object_body_name}'> found")

    return tree, changed_pos, inserted_fj, warnings


def main():
    ap = argparse.ArgumentParser(
        description="Batch patch object_sim/*/object.xml: set pos.y and insert freejoint after <include file='body.xml'/>."
    )
    ap.add_argument(
        "object_sim_dir",
        type=str,
        help="object_sim 目录路径（例如: ./assets/dual_arm_and_single_arm/objects/object_sim）"
    )
    ap.add_argument("--y", type=float, default=0.6, help="把 pos 的 y 坐标改成多少（默认 0.6）")
    ap.add_argument("--dry-run", action="store_true", help="只打印将修改哪些文件，不写回")
    ap.add_argument("--no-backup", action="store_true", help="不生成 .bak 备份（默认会备份）")
    ap.add_argument("--add-pos-if-missing", action="store_true", help="若 body 没有 pos，则补一个默认 pos")
    ap.add_argument("--object-body-name", type=str, default="object", help="要修改的 body 名字（默认 object）")
    ap.add_argument("--include-file", type=str, default="body.xml", help="include 文件名（默认 body.xml）")
    ap.add_argument("--freejoint-name", type=str, default="object_free", help="freejoint name（默认 object_free）")

    args = ap.parse_args()

    object_sim_dir = Path(args.object_sim_dir).resolve()
    if not object_sim_dir.exists():
        raise SystemExit(f"[ERROR] object_sim_dir not found: {object_sim_dir}")

    # 只匹配 “每个子文件夹下的 object.xml”：object_sim/*/object.xml
    targets = sorted(object_sim_dir.glob("*/object.xml"))
    print(f"[INFO] object_sim_dir: {object_sim_dir}")
    print(f"[INFO] found {len(targets)} targets (pattern: */object.xml)")

    n_modified = 0
    n_pos = 0
    n_fj = 0
    n_warn = 0

    for xml_path in targets:
        try:
            tree, changed_pos, inserted_fj, warnings = patch_one_object_xml(
                xml_path,
                args.y,
                object_body_name=args.object_body_name,
                include_file_name=args.include_file,
                freejoint_name=args.freejoint_name,
                add_pos_if_missing=args.add_pos_if_missing,
            )
        except Exception as e:
            print(f"[SKIP] parse/patch failed: {xml_path} | {type(e).__name__}: {e}")
            continue

        if warnings:
            n_warn += len(warnings)
            for w in warnings[:5]:
                print(f"[WARN] {xml_path}: {w}")
            if len(warnings) > 5:
                print(f"[WARN] {xml_path}: ... ({len(warnings)-5} more warnings)")

        if not (changed_pos or inserted_fj):
            # 无变化
            continue

        print(f"[PATCH] {xml_path} | pos_changed={changed_pos} freejoint_inserted={inserted_fj}")

        if args.dry_run:
            continue

        # 写回前备份
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

        n_modified += 1
        n_pos += int(changed_pos)
        n_fj += int(inserted_fj)

    print("\n========== SUMMARY ==========")
    print(f"modified_files: {n_modified} (dry-run 写回为0)")
    print(f"pos_changed_files: {n_pos}")
    print(f"freejoint_inserted_files: {n_fj}")
    print(f"warnings_total: {n_warn}")
    print(f"dry_run: {args.dry_run}")


if __name__ == "__main__":
    main()