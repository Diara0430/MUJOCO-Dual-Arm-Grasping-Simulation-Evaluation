from pathlib import Path
import xml.etree.ElementTree as ET

"""
目的：
生成 dual_ur5_scene.xml：把 dual_ur5.xml 里所有 <mesh file="..."> 的路径统一加前缀 "meshes/"

规则：
- 如果 file 已经以 "meshes/" 开头，保持不变
- 如果 file 是绝对路径（以 / 开头），保持不变
- 否则：file = "meshes/" + file

这样 scene 加载时不再依赖 <compiler meshdir="..."> 是否生效。
"""

def patch_dual_ur5(input_xml: Path, output_xml: Path):
    tree = ET.parse(input_xml)
    root = tree.getroot()

    count = 0
    for mesh in root.iter("mesh"):
        file_attr = mesh.get("file")
        if not file_attr:
            continue

        if file_attr.startswith("/") or file_attr.startswith("meshes/"):
            continue

        mesh.set("file", "meshes/" + file_attr)
        count += 1

    tree.write(output_xml, encoding="utf-8", xml_declaration=True)
    print(f"[OK] patched meshes: {count}")
    print(f"[OK] output: {output_xml}")

def sanity_check(output_xml: Path, base_dir: Path):
    """
    检查 patch 后的 mesh 文件是否真的存在
    base_dir: dual_arm_and_single_arm 目录
    """
    tree = ET.parse(output_xml)
    root = tree.getroot()

    missing = []
    for mesh in root.iter("mesh"):
        f = mesh.get("file")
        if not f:
            continue
        if f.startswith("/"):
            p = Path(f)
        else:
            p = base_dir / f
        if not p.exists():
            missing.append(str(p))

    if missing:
        print("[WARN] missing mesh files:")
        for m in missing[:50]:
            print("  ", m)
        if len(missing) > 50:
            print("  ...", len(missing) - 50, "more")
    else:
        print("[OK] all mesh files exist.")

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    base_dir = repo_root / "assets" / "dual_arm_and_single_arm"

    input_xml = base_dir / "dual_ur5.xml"
    output_xml = base_dir / "dual_ur5_scene.xml"

    patch_dual_ur5(input_xml, output_xml)
    sanity_check(output_xml, base_dir)
