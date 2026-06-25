"""kbmap CLI 入口。

子命令：scan / verify / build-centroids / classify / assign
通过环境变量 KBMAP_EMBEDDER=mock 切换到 MockCloudEmbeddingService（测试用）。
"""
import argparse
import os
import sys
from pathlib import Path

def _make_embedder():
    """根据 KBMAP_EMBEDDER 环境变量返回 embedder。默认真实 CloudEmbeddingService。"""
    if os.environ.get("KBMAP_EMBEDDER") == "mock":
        from app.services.cloud_embedding_service import MockCloudEmbeddingService
        return MockCloudEmbeddingService(dimension=1024)
    from app.services.cloud_embedding_service import CloudEmbeddingService
    try:
        return CloudEmbeddingService()
    except Exception as e:
        sys.stderr.write(f"[kbmap] 无法初始化 CloudEmbeddingService：{e}\n")
        sys.stderr.write("[kbmap] 测试可设 KBMAP_EMBEDDER=mock 用 Mock。\n")
        raise


def cmd_scan(args):
    from app.kbmap.scanner import scan_kb_dirs
    from app.kbmap.manifest import build_manifest_from_inventory, save_manifest, validate_manifest
    root = Path(args.root)
    inv = scan_kb_dirs(root)
    m = build_manifest_from_inventory(inv)
    validate_manifest(m)
    save_manifest(m, Path(args.out))
    print(f"[kbmap] scan 完成：{len(m.files)} 个文件 → {args.out}")
    if args.draft:
        _write_draft_md(m, inv, Path(args.draft))


def _write_draft_md(manifest, inventory, path):
    """写人类可读草案 markdown（供用户编辑）。"""
    from collections import Counter
    cnt = Counter(f.kb for f in manifest.files)
    lines = ["# KB 分类体系草案（待用户编辑）", "",
             "## 1. KB 清单", "| KB | 文件数 | 描述 |",
             "|----|--------|------|"]
    for kb in sorted(manifest.kb_defines):
        lines.append(f"| {kb} | {cnt.get(kb, 0)} | {manifest.kb_defines[kb].description} |")
    lines += ["", "## 2. 去重记录（同 content-hash 被合并）"]
    if not inventory.duplicates:
        lines.append("无去重。")
    else:
        lines += ["| 保留 | 被丢弃（同内容） |", "|------|------------------|"]
        for main, dups in sorted(inventory.duplicates.items()):
            lines.append(f"| {main} | {', '.join(dups)} |")
    lines += ["", "## 3. 含混文件标记（需用户裁决）",
              "- 请用 `python -m app.kbmap verify` 跑验证后参考异常文件清单调整。"]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[kbmap] 草案写入 {path}")


def cmd_verify(args):
    from app.kbmap.manifest import load_manifest, validate_manifest
    from app.kbmap.metrics import verify, write_report_md
    m = load_manifest(Path(args.manifest))
    validate_manifest(m)
    embedder = _make_embedder()
    rep = verify(m, Path(args.root), embedder)
    write_report_md(rep, m, Path(args.out))
    print(f"[kbmap] verify 完成 → {args.out}")
    print(f"[kbmap] 凝聚度: { {k: round(v,3) for k,v in rep.cohesion.items()} }")
    print(f"[kbmap] 异常文件数: {len(rep.outliers)}")


def cmd_build_centroids(args):
    from app.kbmap.manifest import load_manifest, validate_manifest
    from app.kbmap.centroids import build_centroids
    m = load_manifest(Path(args.manifest))
    validate_manifest(m)
    embedder = _make_embedder()
    build_centroids(m, Path(args.root), embedder, Path(args.out_dir))
    print(f"[kbmap] 质心写入 {args.out_dir}")


def cmd_classify(args):
    from app.kbmap.classifier import KBClassifier
    embedder = _make_embedder()
    clf = KBClassifier(Path(args.centroids_dir), embedder)
    res = clf.classify(args.text, top_k=args.top_k)
    for kb, score in res:
        print(f"{kb}\t{score:.4f}")


def cmd_assign(args):
    from app.kbmap.classifier import KBClassifier
    embedder = _make_embedder()
    clf = KBClassifier(Path(args.centroids_dir), embedder)
    res = clf.assign_file(args.path, Path(args.root), top_k=args.top_k)
    for kb, score in res:
        print(f"{kb}\t{score:.4f}")


def build_parser():
    p = argparse.ArgumentParser(prog="python -m app.kbmap", description="KB 分类工具链")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan", help="扫描目录生成 manifest")
    s.add_argument("--root", required=True, help="merged/ 根目录")
    s.add_argument("--out", required=True, help="输出 manifest yaml 路径")
    s.add_argument("--draft", help="（可选）输出人类可读草案 md 路径")
    s.set_defaults(func=cmd_scan)

    v = sub.add_parser("verify", help="跑分类验证")
    v.add_argument("--manifest", required=True)
    v.add_argument("--root", required=True)
    v.add_argument("--out", required=True, help="输出报告 md 路径")
    v.set_defaults(func=cmd_verify)

    b = sub.add_parser("build-centroids", help="构建 KB 质心")
    b.add_argument("--manifest", required=True)
    b.add_argument("--root", required=True)
    b.add_argument("--out-dir", required=True)
    b.set_defaults(func=cmd_build_centroids)

    c = sub.add_parser("classify", help="分类一段文本")
    c.add_argument("text", help="要分类的文本（如用户问题）")
    c.add_argument("--centroids-dir", required=True)
    c.add_argument("--top-k", type=int, default=3)
    c.set_defaults(func=cmd_classify)

    a = sub.add_parser("assign", help="分类单个文件")
    a.add_argument("path", help="相对 root 的文件路径")
    a.add_argument("--root", required=True)
    a.add_argument("--centroids-dir", required=True)
    a.add_argument("--top-k", type=int, default=3)
    a.set_defaults(func=cmd_assign)

    return p


def main(argv=None):
    p = build_parser()
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
