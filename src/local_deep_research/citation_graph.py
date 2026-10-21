"""Citation-graph snowballing for deep research: seeds -> backward+forward expansion -> ranked reading list.

usage: citation_graph.py <research_dir> [--query Q] [--depth 2] [--beam 40] [--top 60] [--seed ID ...] [--deadline 90]

Seeds are the arXiv IDs / DOIs found in <research_dir>/notes/*.md and leads/*.md (plus --seed). Each layer pulls
references (backward) and citations (forward) from the Semantic Scholar batch API (no key; OpenAlex is unusable
without one since its 2026 per-IP budget), keeps the `beam` best neighbours (link count to the current core, then
relevance) and expands them. Final score = relevance² (bge-m3 cosine) x (0.5 + 0.5 x structure: links to the core, PageRank,
citations, recency), so central but off-topic hubs drop. Writes <research_dir>/citation/reading_list.{md,csv} and graph.json. No LLM tokens are used.
HTTP responses are cached in <cache dir>/s2.sqlite (see config.py) so later research reuses them.
Keyless S2 throttles for minutes at a time: all network work shares one wall-clock --deadline (default 90 s, under the
agents' 120 s shell timeout) and at most 3 tries per batch. When throttled or out of time it stops calling S2, writes
a partial citation/reading_list.md that says so (cached + already fetched papers, ranked by graph structure only) and
exits 0.
"""
import argparse
import csv
import json
import math
import pathlib
import re
import sqlite3
import sys
import time

import httpx
import networkx as nx
import numpy as np

S2 = "https://api.semanticscholar.org/graph/v1/paper/batch"
LINK_FIELDS = "title,year,citationCount,externalIds,references.paperId,citations.paperId"
META_FIELDS = "title,year,citationCount,externalIds,abstract,url,openAccessPdf,venue"
from . import config as _config  # noqa: E402
CACHE = _config.load().cache_path() / "s2.sqlite"
ID_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})|\b(10\.\d{4,9}/[^\s)\]>\"'`,;]+)", re.I)


_sleep = time.sleep  # tests replace it


class S2Client:
    TRIES = 3

    def __init__(self, deadline=90.0, http=None):
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(CACHE, timeout=60)  # busy timeout: parallel research share the cache
        self.db.execute("create table if not exists paper (key text primary key, data text)")
        self.http = http or httpx.Client(timeout=30)
        self.calls, self.end, self.throttled, self.why = 0, time.monotonic() + deadline, False, ""

    def close(self):
        """release the cache file (Windows cannot delete or replace a SQLite file another handle keeps open)"""
        self.db.close()
        if hasattr(self.http, "close"):
            self.http.close()

    def left(self):
        return self.end - time.monotonic()

    def stop(self, why):
        if not self.throttled:
            print(f"  S2 stopped: {why}", file=sys.stderr)
        self.throttled, self.why = True, self.why or why

    def batch(self, ids, fields):
        """Return {requested_id: paper|None}; cached per (id, fields). Once throttled / out of time only the cache
        answers (missing ids are simply absent)."""
        out, todo = {}, []
        for i in ids:
            row = self.db.execute("select data from paper where key=?", (f"{fields}|{i}",)).fetchone()
            if row:
                out[i] = json.loads(row[0])
            else:
                todo.append(i)
        for k in range(0, len(todo), 400):
            chunk, r = todo[k:k + 400], None
            for attempt in range(self.TRIES):
                if self.throttled or self.left() < 3:
                    self.stop(self.why or "deadline reached"); break
                try:
                    r = self.http.post(S2, params={"fields": fields}, json={"ids": chunk},
                                       timeout=max(3.0, min(30.0, self.left())))
                except httpx.HTTPError as e:
                    r = None; print(f"  S2 request failed: {str(e)[:80]}", file=sys.stderr)
                self.calls += 1
                if r is not None and r.status_code == 200:
                    break
                if r is not None and r.status_code not in (429,) and r.status_code < 500:
                    r.raise_for_status()
                r = None
                if attempt + 1 < self.TRIES:
                    _sleep(max(0.0, min(5 * 2 ** attempt, self.left() - 3)))
            if r is None:
                self.stop(self.why or f"throttled (429/5xx {self.TRIES}x)")
                break
            for i, p in zip(chunk, r.json()):
                out[i] = p
                self.db.execute("insert or replace into paper values (?,?)", (f"{fields}|{i}", json.dumps(p)))
            self.db.commit()
            _sleep(1.1)  # unauthenticated pool: stay around 1 req/s
        return out


def seeds_from_files(root):
    ids = set()
    for f in list((root / "notes").glob("*.md")) + list((root / "leads").glob("*.md")):
        for arx, doi in ID_RE.findall(f.read_text(encoding="utf-8", errors="ignore")):
            ids.add(f"ARXIV:{arx}" if arx else f"DOI:{doi.rstrip('.')}")
    return sorted(ids)


def question_from_files(root):
    for name in ("PLAN.md", "BRIEF.md"):
        p = root / name
        if p.exists():
            m = re.search(r"\*\*Soru:\*\*\s*(.+)", p.read_text(encoding="utf-8"))
            if m:
                return m.group(1).strip()
    sys.exit("no --query given and no '**Soru:**' line in PLAN.md/BRIEF.md")


def embed(texts):
    from .rt_embed import embed as _embed, pick
    model, dev = pick()
    return _embed(texts, model, dev), f"{model.split('/')[-1]} {dev}"


def norm(v):
    v = np.asarray(v, dtype=float)
    return (v - v.min()) / (v.max() - v.min()) if v.max() > v.min() else np.zeros_like(v)


def partial_note(s2):
    return (f"**KISMİ LİSTE: Semantic Scholar {s2.why}.** Yalnız önbellekteki / çekilebilen makaleler; benzerlik "
            "hesaplanmadı, sıralama yalnız graf yapısına göre. Önbellek korunur: daha sonra tekrar çalıştır.\n")


def write_empty(root, query, s2, n_ids):
    out = root / "citation"; out.mkdir(exist_ok=True)
    (out / "reading_list.md").write_text(f"# Okuma listesi (atıf grafiği)\n\nSoru: {query}\n\n{partial_note(s2)}\n"
                                         f"Tohum kimliği: {n_ids}, çözülen: 0, S2 çağrısı: {s2.calls}.\n", encoding="utf-8")
    print(f"wrote {out}/reading_list.md (partial, S2 {s2.why}); exit 0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=pathlib.Path)
    ap.add_argument("--query")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--beam", type=int, default=40)
    ap.add_argument("--top", type=int, default=60)
    ap.add_argument("--seed", nargs="*", default=[])
    ap.add_argument("--deadline", type=float, default=90, help="wall-clock seconds for all S2 calls")
    a = ap.parse_args()
    root = a.root.resolve()
    query = a.query or question_from_files(root)
    s2 = S2Client(a.deadline)
    try:
        return _run(a, root, query, s2)
    finally:
        s2.close()  # every exit path (early returns included) releases the SQLite cache


def _run(a, root, query, s2):

    seed_ids = sorted(set(seeds_from_files(root)) | set(a.seed))
    resolved = s2.batch(seed_ids, LINK_FIELDS)
    papers = {}  # paperId -> link-level record
    seeds = set()
    for p in resolved.values():
        if p and p.get("paperId"):
            papers[p["paperId"]] = p
            seeds.add(p["paperId"])
    missing = [i for i in seed_ids if not resolved.get(i)]
    print(f"seeds: {len(seed_ids)} ids -> {len(seeds)} resolved in Semantic Scholar"
          + (f"; not found: {', '.join(missing[:8])}" if missing else ""))
    if not seeds:
        if s2.throttled:
            write_empty(root, query, s2, len(seed_ids)); return
        sys.exit("no seed resolved")

    G = nx.DiGraph()  # citing -> cited
    core, frontier = set(seeds), set(seeds)
    for layer in range(1, a.depth + 1):
        neigh = {}
        for pid in frontier:
            p = papers.get(pid) or {}
            for r in p.get("references") or []:
                if r.get("paperId"):
                    G.add_edge(pid, r["paperId"]); neigh.setdefault(r["paperId"], set()).add(pid)
            for c in p.get("citations") or []:
                if c.get("paperId"):
                    G.add_edge(c["paperId"], pid); neigh.setdefault(c["paperId"], set()).add(pid)
        cand = sorted((n for n in neigh if n not in core), key=lambda n: -len(neigh[n]))[: a.beam * 3]
        got = s2.batch(cand, LINK_FIELDS)
        for n, p in got.items():
            if p:
                papers[n] = p
        # beam: most connected to the current core first; ties broken by citations
        nxt = sorted((n for n in cand if papers.get(n)),
                     key=lambda n: (-len(neigh[n]), -(papers[n].get("citationCount") or 0)))[: a.beam]
        core |= set(nxt); frontier = set(nxt)
        print(f"layer {layer}: {len(neigh)} neighbours, {len(cand)} fetched, {len(nxt)} expanded; graph {G.number_of_nodes()} nodes")
        if s2.throttled:
            break

    # final ranking over every fetched paper
    ids = [n for n in papers if papers[n].get("title")]
    meta = s2.batch(ids, META_FIELDS) if not s2.throttled else {}
    partial = s2.throttled
    texts = [f"{(meta.get(n) or {}).get('title') or papers[n]['title']}. {(meta.get(n) or {}).get('abstract') or ''}"
             for n in ids]
    print(f"abstracts: {sum(1 for n in ids if (meta.get(n) or {}).get('abstract'))}/{len(ids)}")
    if partial:  # throttled: no embedding model load, finish fast (the caller's shell timeout is ~120 s)
        rel, dev = np.ones(len(ids)), "yok (kısmi)"
    else:
        vecs, dev = embed([query] + texts)
        rel = vecs[1:] @ vecs[0]
    pr = nx.pagerank(G) if G.number_of_nodes() else {}
    links =[sum(1 for m in set(G.predecessors(n)) | set(G.successors(n)) if m in core) if n in G else 0 for n in ids]
    cites = [math.log1p(papers[n].get("citationCount") or 0) for n in ids]
    years = [papers[n].get("year") or 2000 for n in ids]
    recency = [min(1.0, max(0.0, (y - 2015) / 10)) for y in years]
    # structure only amplifies relevance: generic hubs (Attention, LSTM, GPT-3) are central but off-topic
    struct = 0.45 * norm(links) + 0.30 * norm([pr.get(n, 0) for n in ids]) + 0.10 * norm(cites) \
        + 0.15 * np.asarray(recency)
    score = (norm(rel) ** 2 if not partial else 1.0) * (0.5 + 0.5 * struct)

    order = np.argsort(-score)
    out = root / "citation"; out.mkdir(exist_ok=True)
    rows = []
    for i in order:
        n = ids[i]; m = meta.get(n) or {}; ext = papers[n].get("externalIds") or {}
        pdf = (m.get("openAccessPdf") or {}).get("url") or ""
        rows.append({
            "rank": len(rows) + 1, "score": round(float(score[i]), 3), "relevance": round(float(rel[i]), 3),
            "links_to_core": links[i], "seed": n in seeds, "year": papers[n].get("year"),
            "citations": papers[n].get("citationCount"), "title": m.get("title") or papers[n].get("title"),
            "arxiv": ext.get("ArXiv", ""), "doi": ext.get("DOI", ""), "pdf": pdf,
            "url": m.get("url") or f"https://www.semanticscholar.org/paper/{n}", "s2_id": n,
        })
    if not rows:
        write_empty(root, query, s2, len(seed_ids)); return
    with open(out / "reading_list.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    new = [r for r in rows if not r["seed"]][: a.top]
    md = [f"# Okuma listesi (atıf grafiği)\n", f"Soru: {query}\n"] + ([partial_note(s2)] if partial else []) + [
          f"Tohum: {len(seeds)} makale, graf: {G.number_of_nodes()} düğüm / {G.number_of_edges()} kenar, "
          f"S2 çağrısı: {s2.calls}, embedding: {dev}.\n",
          "Puan = benzerlik² × (0.5 + 0.5 × yapı); yapı = 0.45 çekirdeğe bağlantı + 0.30 PageRank + 0.10 atıf + 0.15 yenilik "
          "(konu dışı ama merkezi makaleler elenir). "
          "Tohumlar hariç ilk adaylar:\n",
          "| # | puan | benz. | bağ | yıl | atıf | başlık | link |", "|---|---|---|---|---|---|---|---|"]
    for k, r in enumerate(new, 1):
        link = r["pdf"] or (f"https://arxiv.org/abs/{r['arxiv']}" if r["arxiv"] else r["url"])
        md.append(f"| {k} | {r['score']} | {r['relevance']} | {r['links_to_core']} | {r['year']} | {r['citations']} | "
                  f"{(r['title'] or '').replace('|', '/')} | {link} |")
    (out / "reading_list.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    with open(out / "graph.json", "w", encoding="utf-8") as f:
        json.dump({"nodes": rows, "edges": [[u, v] for u, v in G.edges() if u in papers and v in papers]}, f)
    print(f"wrote {out}/reading_list.md ({len(new)} candidates{', PARTIAL: S2 ' + s2.why if partial else ''}), "
          f"reading_list.csv ({len(rows)}), graph.json; S2 calls {s2.calls}")


if __name__ == "__main__":
    main()
