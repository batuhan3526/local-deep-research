"""Offline checks for v3.2 (context budget): char-capped pack parts, the code-built outline (outline.py: every
cluster in exactly one sub-topic, model mistakes repaired, Yüksek never appendix-only, chunk cache), claims_by_id
--topic, sections.py and verify_fix.py. The opencode read permissions are checked in suite_legacy_opencode.py.
The model is faked: no LLM, no network (CPU embeddings from the local HF cache).
usage: python tests/suite_v32.py [scratch_dir]"""
import contextlib
import io
import json
import re
import shutil

from _suite import finish, scratch  # noqa: E402  (sets hermetic LDR_* env first)

SCRATCH = scratch("v32-test-")

from local_deep_research import claims_by_id  # noqa: E402
from local_deep_research import harvest  # noqa: E402
from local_deep_research import outline  # noqa: E402
RESULTS = [0, 0]


def check(name, cond, info=""):
    RESULTS[0 if cond else 1] += 1
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  -> {str(info)[:400]}"))


# ---------- 1. pack parts capped by chars ----------
sp = harvest.split_pack(["y" * 500] * 400)
check("split_pack: 400 lines of 500 chars -> parts of <= PACK_CHARS (was one 200k-char file = 53k tokens)",
      len(sp) == 4 and all(sum(len(x) + 1 for x in p) <= harvest.PACK_CHARS for p in sp) and sum(map(len, sp)) == 400,
      [len(p) for p in sp])
check("split_pack: line cap still applies, empty pack -> one empty part, one huge line gets its own part",
      [len(p) for p in harvest.split_pack(["x"] * 905, 400, 10 ** 9)] == [400, 400, 105] and harvest.split_pack([]) == [[]]
      and [len(p) for p in harvest.split_pack(["a", "b" * 100, "c"], 400, 50)] == [1, 1, 1])

# ---------- 2. outline.py with a faked model ----------
R = SCRATCH / "t/research"; (R / "claims").mkdir(parents=True); (R / "harvest").mkdir()
(R / "harvest/plan.json").write_text(json.dumps({"subquestions": [{"id": 2, "title": "Features", "text": "which features"}]}), encoding="utf-8")
(R / "harvest/profile.json").write_text(json.dumps({"out_lang": "en"}), encoding="utf-8")
TOPICS = ["momentum signal return", "volatility regime forecast", "purged cross validation embargo"]


def line(k):
    lv = "Yüksek" if k <= 6 else "Orta"
    return (f"- **{lv}** ({2 if lv == 'Yüksek' else 1} alan) claim {k} about {TOPICS[k % 3]} — \"q{k}\" "
            f"[https://a{k}.org/x] ⟨S2-{k:04d}⟩")


(R / "claims/subq-2.md").write_text("# Kanıt paketi S2\n\n" + "\n".join(line(k) for k in range(1, 41)) + "\n", encoding="utf-8")
(R / "claims/subq-2-p2.md").write_text("# part 2\n\n" + "\n".join(line(k) for k in range(41, 61)) + "\n", encoding="utf-8")
CALLS = []


def fake_llm(prompt, max_tokens, kind):
    CALLS.append(kind)
    ids = re.findall(r"^(S2-\d{4}) \[", prompt, re.M)
    if kind == "outline-themes":
        # mistakes on purpose: one id skipped, one invented, one repeated in two themes
        by = {t: [i for i in ids if TOPICS[int(i[3:]) % 3] == t] for t in TOPICS}
        skipped = by[TOPICS[0]].pop()
        return {"themes": [{"title": t, "summary": f"about {t}", "ids": v + (["S2-9999", by[TOPICS[1]][0]] if t == TOPICS[2] else [])}
                           for t, v in by.items() if v], "_skipped": skipped}
    keys = re.findall(r"^(c\d+t\d+) \|", prompt, re.M)
    vol = [k for k in keys if "volatility" in prompt.split(k + " |", 1)[1].split("\n", 1)[0]]
    rest = [k for k in keys if k not in vol]
    # the volatility themes (they hold Yüksek ids) go to the appendix, one theme is forgotten
    return {"subtopics": [{"title": "Signals", "summary": "momentum and cv", "themes": rest[:-1]}], "appendix_only": vol}


outline.llm = fake_llm
outline.CHUNK_CHARS = 1500  # several chunks from 60 short lines
with contextlib.redirect_stdout(io.StringIO()):
    outline.main([str(R)])
o = json.loads((R / "analysis/s2-outline.json").read_text(encoding="utf-8"))
placed = [i for s in o["subtopics"] for i in s["ids"]] + [i for a in o["appendix_only"] for i in a["ids"]]
check("outline: every id placed exactly once despite skipped / invented / repeated ids and a forgotten theme",
      sorted(placed) == [f"S2-{k:04d}" for k in range(1, 61)], (len(placed), len(set(placed))))
high = {f"S2-{k:04d}" for k in range(1, 7)}
check("outline: Yüksek clusters never end up appendix-only (moved to the closest sub-topic)",
      not high & {i for a in o["appendix_only"] for i in a["ids"]} and o["appendix_only"], o["appendix_only"])
s1 = o["subtopics"][0]
check("outline: sub-topic keys S<n>.<k>, writer ids start with every Yüksek one, md written",
      s1["key"] == "S2.1" and set(s1["writer_ids"][:6]) == high and (R / "analysis/s2-outline.md").exists()
      and "## S2.1 Signals" in (R / "analysis/s2-outline.md").read_text(encoding="utf-8"), s1["writer_ids"][:8])
n_theme = CALLS.count("outline-themes")
CALLS.clear()
with contextlib.redirect_stdout(io.StringIO()):
    outline.main([str(R)])
check("outline: a restart re-asks nothing (chunk + merge cache): sub-topic keys stay stable",
      n_theme > 1 and CALLS == [] and json.loads((R / "analysis/s2-outline.json").read_text(encoding="utf-8"))["subtopics"] == o["subtopics"],
      (n_theme, CALLS))
outline.llm = lambda *a: (_ for _ in ()).throw(ValueError("down"))
shutil.rmtree(R / "analysis")
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    outline.main([str(R)])
o2 = json.loads((R / "analysis/s2-outline.json").read_text(encoding="utf-8"))
check("outline: model down -> still every id placed (one 'Unsorted' theme per chunk)",
      sorted(i for s in o2["subtopics"] for i in s["ids"]) == [f"S2-{k:04d}" for k in range(1, 61)], o2["subtopics"][:1])

# ---------- 3. claims_by_id --topic ----------
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    rc = claims_by_id.main(["x", str(R), "--topic", "S2.1"])
out = buf.getvalue().splitlines()
check("claims_by_id --topic: the sub-topic's writer lines, Yüksek first; unknown key -> exit 1",
      rc == 0 and len(out) == len(o2["subtopics"][0]["writer_ids"]) and "**Yüksek**" in out[0]
      and claims_by_id.main(["x", str(R), "--topic", "S9.9"]) == 1, out[:2])

# ---------- 3b. sections.py (phases 4-5 by code) with a faked model ----------
from local_deep_research import sections  # noqa: E402
txt, rem = sections.keep_urls("see https://a1.org/x and https://evil.example/p. Also [paper](https://fake.org/z) "
                              "and [ok](https://a2.org/x)", {"https://a1.org/x", "https://a2.org/x"})
check("sections.keep_urls: URLs not in the evidence lines are removed (link text kept), allowed ones stay",
      "https://a1.org/x" in txt and "evil.example" not in txt and "fake.org" not in txt and "paper" in txt
      and "https://a2.org/x" in txt and sorted(rem) == ["https://evil.example/p", "https://fake.org/z"], (txt, rem))
SCALLS = []


def fake_sec(prompt, max_tokens, kind):
    SCALLS.append(kind)
    if kind == "section-analyze":
        key = re.search(r"Sub-topic (S\d+\.\d+)", prompt).group(1)
        return (f"### {key} T\n\n#### Key findings\n- [High] (i) x - https://a1.org/x ⟨S2-0001⟩\n"
                f"- [Medium] (ii) y - https://invented.org/q ⟨S2-0009⟩\n")
    if kind == "section-summary":
        return "## Özet\n\n- [High] answer S2.1 - https://a1.org/x\n"
    if kind == "section-write":
        return ("<think>plan</think># Big title\n\nParagraph with https://a1.org/x ⟨S2-0001⟩ and https://nope.org/1.\n\n"
                "## Sources\n\n1. https://nope.org/1\n")
    return "# Features\n\nIntro paragraph."


sections.llm = fake_sec
with contextlib.redirect_stdout(io.StringIO()):
    sections.main(["analyze", str(R)])
an = (R / "analysis/s2.md").read_text(encoding="utf-8")
check("sections analyze: analysis/s2.md = Özet + one block per sub-topic, invented URLs removed",
      an.count("### S2.") == len(o2["subtopics"]) and "## Özet" in an and "https://a1.org/x" in an
      and "invented.org" not in an, an[:300])
n_an = SCALLS.count("section-analyze"); SCALLS.clear()
with contextlib.redirect_stdout(io.StringIO()):
    sections.main(["analyze", str(R)])
check("sections analyze: a restart re-asks nothing (cache)", n_an == len(o2["subtopics"]) and SCALLS == [], SCALLS)
with contextlib.redirect_stdout(io.StringIO()):
    try:
        sections.main(["write", str(R)]); no_style = False
    except SystemExit:
        no_style = True
check("sections write: refuses without STYLE.md (the merge analyst writes it first)", no_style)
(R / "BRIEF.md").write_text("# BRIEF\nETF direction project", encoding="utf-8")
(R / "STYLE.md").write_text("# Style\n- IC, not ic", encoding="utf-8")
with contextlib.redirect_stdout(io.StringIO()):
    rc = sections.main(["write", str(R)])
secs = sorted(p.name for p in (R / "report").glob("02-*.md"))
body = (R / "report" / [x for x in secs if x.startswith("02-01-")][0]).read_text(encoding="utf-8")
check("sections write: NN-00 intro + NN-MM per sub-topic; '#' demoted to '##', think stripped, code-built Sources "
      "list with only evidence URLs", rc == 0 and secs[0].startswith("02-00-") and len(secs) == 1 + len(o2["subtopics"])
      and body.startswith("## Big title") and "<think>" not in body and "nope.org" not in body
      and body.count("## Sources") == 1 and "1. https://a1.org/x" in body, (secs, body))

# ---------- 3c. verify_fix.py: edits are exact, unique and cannot invent ----------
from local_deep_research import verify_fix  # noqa: E402
src = "Alpha is 0.52 per https://a1.org/x ⟨S2-0001⟩. Beta rose. Beta rose."
new, ap, rj = verify_fix.apply_edits(src, [
    {"old": "Alpha is 0.52", "new": "Alpha is 0.52 (by analogy from medicine)", "why": "label"},
    {"old": "Beta rose.", "new": "Beta rose 3%.", "why": "dup old"},
    {"old": "Alpha is", "new": "Alpha is 0.99, see https://evil.org/x and", "why": "invent"},
    {"old": "not there", "new": "x", "why": "missing"},
    {"old": "per https://a1.org/x", "new": "per https://a1.org/x and https://a2.org/x", "why": "allowed url"}],
    {"https://a1.org/x", "https://a2.org/x"}, {"0.52"})
check("verify_fix.apply_edits: exact unique edits applied; repeated / missing old text, new URLs and new numbers rejected",
      "(by analogy from medicine)" in new and "a2.org" in new and "evil" not in new and "0.99" not in new
      and len(ap) == 2 and sorted(r[3].split()[0] for r in rj) == ["new", "old", "old"], (new, [r[3] for r in rj]))
(R / "claims/local-facts.md").write_text("- DEFINITION: mean |IC| 0.235 -> 0.216 = -8.5% relative\n", encoding="utf-8")
VCALLS = []


def fake_verify(prompt, max_tokens):
    VCALLS.append(prompt)
    sec = prompt.split("<<<SECTION", 1)[1].split("SECTION>>>", 1)[0]
    edits = [{"old": "Paragraph with", "new": "Paragraph (mean |IC| 0.235 -> 0.216, -8.5% relative) with", "why": "local"}] \
        if "Paragraph with" in sec else []
    return {"checks": [{"claim": "x", "verdict": "SUPPORTED-1", "evidence": "⟨S2-0001⟩"}, {"claim": "y", "verdict": "BOGUS"}],
            "edits": edits}


verify_fix.llm = fake_verify
(R / "report/02-01-x.md").write_text((R / "report").glob("02-01-*.md").__next__().read_text(encoding="utf-8"), encoding="utf-8")  # a second topic file
with contextlib.redirect_stdout(io.StringIO()):
    verify_fix.main([str(R)])
v2 = (R / "VERIFY-02.md").read_text(encoding="utf-8")
sec1 = [p for p in sorted((R / "report").glob("02-01-*.md"))][0].read_text(encoding="utf-8")
check("verify_fix: VERIFY-02.md (valid verdicts only, applied edits), one 95-dogrulama.md section, edit applied with the local "
      "numbers, Sources list rebuilt once",
      "SUPPORTED-1" in v2 and "BOGUS" not in v2 and "## Applied edits" in v2 and (R / "report/95-dogrulama.md").exists() and not (R / "report/02-dogrulama.md").exists()
      and "0.235 -> 0.216" in sec1 and sec1.count("## Sources") == 1, (v2[:300], sec1[:300]))
n_v = len(VCALLS); VCALLS.clear()
with contextlib.redirect_stdout(io.StringIO()):
    verify_fix.main([str(R)])
check("verify_fix: a re-run skips sections unchanged since their pass (cache on the edited text)", n_v >= 3 and VCALLS == [],
      (n_v, len(VCALLS)))

finish(RESULTS, SCRATCH)
