"""Assemble research/report/NN-*.md section files into research/REPORT.md (sorted by file name).

usage: python -m local_deep_research.assemble_report <research_dir>

Top-level numbered headings ("# 6. Doğrulama", "# 06 — Modeller") are renumbered sequentially in file order, so a
late 06-dogrulama.md next to 06-karsi-gorusler.md cannot give two "# 6." sections (a live run). The first
numbered heading keeps its number (0- or 1-based) and zero-padding; separator and heading text are kept; headings
inside ``` fences and "##" subsections are left alone. Section files themselves are never rewritten.
"""
import pathlib
import re
import sys

_NUM_HEAD = re.compile(r"^#[ \t]+(\d{1,3})([ \t]*(?:[.)]|[—–-])[ \t]+)(\S.*)$")  # not "# 1.5 x", "# 186 ETF"
_FENCE = re.compile(r"^\s*(```|~~~)")


def renumber(texts):
    """[section text] -> ([renumbered text], [(old, new, heading)]) with one running counter over all sections."""
    n, out, changes = None, [], []
    for t in texts:
        lines, fence = t.split("\n"), False
        for i, line in enumerate(lines):
            if _FENCE.match(line):
                fence = not fence
            m = None if fence else _NUM_HEAD.match(line)
            if not m:
                continue
            old, sep, rest = m.groups()
            n = int(old) if n is None else n + 1
            new = str(n).zfill(len(old)) if old.startswith("0") and len(old) > 1 else str(n)
            if new != old:
                changes.append((old, new, rest[:80]))
            lines[i] = f"# {new}{sep}{rest}"
        out.append("\n".join(lines))
    return out, changes


def main(argv):
    """argv as sys.argv: [program, research_dir]"""
    root = pathlib.Path(argv[1] if len(argv) > 1 else "research").resolve()
    if root.name != "research" and (root / "research").is_dir():
        root = root / "research"
    parts = sorted((root / "report").glob("[0-9][0-9]-*.md"))
    if not parts:
        sys.exit(f"no section files in {root / 'report'}")
    texts, changes = renumber([p.read_text(encoding="utf-8").strip() for p in parts])
    (root / "REPORT.md").write_text("\n\n".join(texts) + "\n", encoding="utf-8")
    print(f"REPORT.md <- {len(parts)} sections: {', '.join(p.name for p in parts)}")
    for old, new, head in changes:
        print(f"  renumbered: # {old} -> # {new}  {head}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
