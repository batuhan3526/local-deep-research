"""Local-project numbers, verbatim: research/notes/local-project.md -> research/claims/local-facts.md (v3x, 2026-10-18).

usage: local_facts.py <research_dir>

Every line of the local-project notes that holds a measured or compared number - a percentage, a number with a unit
(5 ms, 12 kg, $3.2M, 40 years), a decimal, an "N = ..." count, or a comparison / change (A -> B, <, >, =, vs) - is
copied as-is (with its line number), so section writers quote the user's own figures instead of paraphrasing them.
A live run read "metric -8.5%" as a negative value when the line meant a change A -> B of -8.5% RELATIVE. No keyword
list: the notes can be about anything. Deterministic, no LLM. The notes are agent-written from local files: lines are
only stripped of control / chat-template tokens (corpus.sanitize, invisible otherwise), never reworded; size and line
caps keep a huge file from flooding the list.
"""
import pathlib
import re
import sys

from .corpus import atomic_write, sanitize, under  # noqa: E402

MAX_BYTES, MAX_LINES, MAX_LINE = 2 << 20, 400, 2000
_D = r"\d+(?:[.,]\d+)*"
# units: SI / time / data / money / scale words; letters that are also words or labels ("in", "A", "B") left out
_UNIT = (r"pp|bps?|x|×|ms|s|sec|secs|seconds?|min|mins|minutes?|h|hrs?|hours?|days?|weeks?|months?|years?|yrs?|"
         r"mg|g|kg|tons?|tonnes?|ml|mm|cm|m|km|miles?|ft|°[CF]?|Hz|kHz|MHz|GHz|W|kW|MW|GW|kWh|MWh|V|"
         r"KB|MB|GB|TB|k|K|M|bn|million|billion|thousand|USD|EUR|GBP|JPY|CNY")
# a fact line: a percentage, a currency amount, a number with a unit, a decimal, a count "N = ...", or a number in a
# comparison / change; a bare integer (a year, "5 times", a section number) alone is not a fact
_FACT = re.compile(rf"%|[$€£¥₺]\s?{_D}|{_D}\s?[$€£¥₺]|(?<![\w.]){_D}\s?(?:{_UNIT})(?![\w])|\d[.,]\d|"
                   rf"\b[Nn]\s?=\s?\d|(?:→|->|⇒|≤|≥|<|>|=|±|Δ|\bvs\.?\b)\s?[−+-]?\d|\d\s?(?:→|->|⇒|≤|≥|<|>|±)")
_ARROW = re.compile(r"→|->|⇒")
HEAD = """# Local project numbers (automatic, verbatim)

Source: notes/local-project.md ({n} lines selected{cut}). Quote these numbers in the report VERBATIM, together with
their definition; never reinterpret, flip the sign or round them. A line of the form "A -> B (-x%)" is usually a
RELATIVE change from A to B: the quantity itself is not negative, and the change is versus A, not versus another
baseline. `L<n>` is the line number in the notes file; "(change)" marks a line with an A -> B change.
"""


def facts(text):
    """[(line_no, line)] of numeric fact lines, original text (sanitised, right-stripped), capped."""
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        line = " ".join(sanitize(line).split())  # one line: a stripped chat token must not leave a line break
        if line.strip() and _FACT.search(line):
            out.append((i, line[:MAX_LINE]))
    return out


def main(argv):
    if len(argv) < 2:
        sys.exit(__doc__)
    root = pathlib.Path(argv[1])
    src = under(root, "notes/local-project.md")
    if not src.is_file():
        print(f"local_facts: no {src} (nothing to do)")
        return 0
    with open(src, "rb") as f:
        raw = f.read(MAX_BYTES + 1)
    fl = facts(raw[:MAX_BYTES].decode("utf-8", "replace"))
    cut = len(raw) > MAX_BYTES or len(fl) > MAX_LINES
    md = [HEAD.format(n=min(len(fl), MAX_LINES), cut=", CUT: first part only" if cut else "")]
    md += [f"- L{i}{' (change)' if _ARROW.search(line) else ''}: {line.strip()}" for i, line in fl[:MAX_LINES]]
    out = under(root, "claims/local-facts.md")
    out.parent.mkdir(exist_ok=True)
    atomic_write(out, "\n".join(md) + "\n")
    print(f"local_facts: {min(len(fl), MAX_LINES)} fact lines -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
