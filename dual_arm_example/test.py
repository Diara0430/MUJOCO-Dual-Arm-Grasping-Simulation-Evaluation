#!/usr/bin/env python3
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

ASSET_TAGS = {
    "mesh": ["file"],
    "texture": ["file", "fileright", "fileleft", "fileup", "filedown", "filefront", "fileback"],
    "hfield": ["file"],
    "skin": ["file"],
    "model": ["file"],
}

def norm(p: Path) -> str:
    try:
        return str(p.resolve())
    except Exception:
        return str(p)

def read_xml(path: Path):
    txt = path.read_text(encoding="utf-8")
    return ET.fromstring(txt)

def is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")

def expand_includes(entry_xml: Path, max_depth=50):
    """
    把 include 递归展开：返回一个“展开后的元素列表”用于扫描 file=...
    注意：我们不真的拼成一个大 XML，只是做递归遍历与定位。
    """
    visited = set()
    stack = [(entry_xml, 0)]
    nodes = []  # list of (xml_path, element)

    while stack:
        cur_path, depth = stack.pop()
        if depth > max_depth:
            raise RuntimeError(f"Include too deep > {max_depth}: {cur_path}")

        cur_path = cur_path.resolve()
        if cur_path in visited:
            # include 规则里同一文件通常只允许一次；这里跳过重复
            continue
        visited.add(cur_path)

        try:
            root = read_xml(cur_path)
        except Exception as e:
            print(f"[ERROR] parse failed: {cur_path}\n  {e}")
            continue

        # 收集当前文件的所有元素
        for el in root.iter():
            nodes.append((cur_path, el))

        # 处理 include
        for inc in root.findall(".//include"):
            f = inc.get("file", "")
            if not f:
                continue
            if is_url(f):
                print(f"[WARN] include url (skip): {f}")
                continue

            inc_path = (cur_path.parent / f).resolve()
            if not inc_path.exists():
                print(f"[MISSING INCLUDE] {cur_path} includes {f} -> {inc_path} (NOT FOUND)")
                continue
            stack.append((inc_path, depth + 1))

    return nodes

def scan_assets(nodes):
    """
    扫描 mesh/texture/hfield/... 的 file=...，并检查文件是否存在。
    """
    missing = []
    found = []
    by_xml = defaultdict(list)

    for xml_path, el in nodes:
        tag = el.tag
        if tag not in ASSET_TAGS:
            continue
        for attr in ASSET_TAGS[tag]:
            f = el.get(attr)
            if not f:
                continue
            if is_url(f):
                continue
            # MuJoCo 的 file 一般按入口/meshdir 等规则找；我们这里用一个“保守检查”：
            # 1) 先按 “该 XML 所在目录”解析一遍（你觉得“原本路径对”的含义）
            # 2) 再按 “入口 XML 所在目录”解析一遍（include 后常见的真实基准）
            # 你可以从输出看到哪个成功、哪个失败
            by_xml[xml_path].append((tag, attr, f))

    return by_xml

def check_paths(entry_xml: Path, by_xml):
    entry_dir = entry_xml.resolve().parent
    missing_records = []

    for xml_path, items in by_xml.items():
        xml_dir = xml_path.resolve().parent
        for tag, attr, f in items:
            p1 = (xml_dir / f).resolve()
            p2 = (entry_dir / f).resolve()
            ok1 = p1.exists()
            ok2 = p2.exists()

            if ok1 or ok2:
                print(f"[OK] {xml_path.name}: <{tag} {attr}='{f}'>"
                      f" | as_xml_dir={ok1} ({p1}) | as_entry_dir={ok2} ({p2})")
            else:
                print(f"[MISSING] {xml_path.name}: <{tag} {attr}='{f}'>"
                      f" | tried {p1} and {p2}")
                missing_records.append((xml_path, tag, attr, f, p1, p2))

    return missing_records

def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_mjcf_assets.py /path/to/scene.xml")
        sys.exit(1)

    entry_xml = Path(sys.argv[1]).resolve()
    if not entry_xml.exists():
        print("Entry xml not found:", entry_xml)
        sys.exit(2)

    print("[INFO] entry:", entry_xml)
    nodes = expand_includes(entry_xml)
    print("[INFO] expanded xml files:", len(set(p for p,_ in nodes)))

    by_xml = scan_assets(nodes)
    missing = check_paths(entry_xml, by_xml)

    print("\n========== SUMMARY ==========")
    print("Total referenced assets:", sum(len(v) for v in by_xml.values()))
    print("Missing assets:", len(missing))
    if missing:
        print("\n[DETAIL] Missing list:")
        for xml_path, tag, attr, f, p1, p2 in missing:
            print(f" - in {xml_path}: <{tag} {attr}='{f}'>")
            print(f"   tried1(xml_dir): {p1}")
            print(f"   tried2(entry_dir): {p2}")

if __name__ == "__main__":
    main()