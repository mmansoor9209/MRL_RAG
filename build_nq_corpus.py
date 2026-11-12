#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_nq_corpus.py

Step 0 of the MCF-RAG rebuild for Natural Questions: construct ONE fixed
retrieval corpus, built exactly once, shared by every downstream module.

This is the NQ counterpart of build_hotpot_corpus_v2.py. It keeps that
script's discipline (startup validation, --inspect_only, --overwrite guard,
corpus_hash provenance, integer gold_ids, diagnostics printed before save)
and changes only what the dataset actually requires.


WHY NQ NEEDS A DIFFERENT BUILDER
--------------------------------
Measured on lighteval/natural_questions_helm before writing any of this:

1. TITLE DEDUP IS INVALID.
   307,373 train examples span only 108,074 unique titles, with 179,870
   title->different-text conflicts (58% of train). HotpotQA had 0.64%.
   Deduplicating by title here would silently discard the majority of the
   text and orphan the golds inside it.
   => Dedup key is the CHUNK TEXT HASH.

2. THE RETRIEVAL UNIT CANNOT BE THE ARTICLE.
   `document` is a full Wikipedia article: p50 26,384 chars, p99 165,566,
   max 523,319. The previous NQ pipeline encoded these at max_length=128
   tokens (~500 chars), so the retriever only ever saw the article's opening
   line -- effectively matching on the title. Its reported R@20 of 96% was
   also measured against a corpus built from the 1,000 eval examples' own
   documents, which is the "corpus derived from the eval subset" failure.
   => We chunk, and the corpus is pooled over all splits, built once.

3. GOLD CANNOT BE LOCATED BY SUBSTRING MATCH.
   `long_answers` is answer-bearing and reliable -- short_answers occurs
   inside long_answers[0] in 1646/1646 checked examples -- but it comes from
   a DIFFERENT text-extraction pass than `document`. Raw substring match
   succeeds 10/1646. Whitespace-insensitive match: 382/1646.
   Full canonicalization (NFKD, strip combining marks, drop [12] citation
   markers, keep alphanumerics only): 90.1%.
   => Gold is located by canonical-offset alignment, mapped back to raw
      character offsets, then intersected with chunk spans.
   The residual ~10% are flattened tables and infoboxes (episode lists,
   navboxes), whose flattening differs between the two extraction passes.
   Those examples are recorded with gold_locatable=False and counted in
   meta -- reported, never silently dropped.

4. FIXED-OFFSET CHUNKING MANGLES GOLDS.
   Measured on 2,000 articles, fraction of golds landing WHOLE in a single
   chunk:
       fixed offset 1000/200 : 55.5%   gold_ids p50 = 2
       fixed offset 2000/400 : 85.8%   gold_ids p50 = 1
       structural   budget 1000 : 92.9%   gold_ids p50/p90 = [1, 1]
   Gold passages ARE paragraphs, so splitting on the document's own
   structural boundaries puts chunk edges where gold edges already are.
   Structural units are p50 202 chars, ~56 per article; a typical gold
   (p50 527 raw chars) is 2-3 of them.
   => Structural packing to a character budget, NOT fixed offsets.
   Beyond budget 1000 the curve is flat (95.5% at 2000), so the small end
   is taken: bigger chunks buy ~2 points and cost generator prompt space.

5. THERE ARE NO DISTRACTORS.
   HotpotQA's distractor config supplies 8 labelled hard negatives per
   example for free. NQ supplies none.
   => neg_ids are the example's own article's non-gold chunks: topically
      adjacent by construction, the closest available analogue.

6. 65% OF TRAIN HAS NO ANSWER.
   short_answers and long_answers are empty on 200,447/307,373 train and
   3,541/7,830 validation (the two fields are empty together -- it is one
   answerability flag, not two). Retrieval can still use these examples;
   generation cannot.
   => Every record carries `answerable`. Nothing is dropped at build time;
      each module filters explicitly and states its pool.


TWO PASSES
----------
Pass 1 (expensive, run once)  -> nq_chunks_full.pt
    Every chunk of every article, deduplicated and deterministically
    ordered. ~11.6M chunks, ~11.7GB. This file is NEVER rebuilt.

Pass 2 (cheap, repeatable)    -> nq_corpus.pt + nq_qa_{split}.pt
    A deterministic subsample of Pass 1 at --corpus_size, retaining ALL gold
    chunks plus a seeded sample of the rest, with its own corpus_hash and
    contiguous integer doc_ids.

This is why the corpus-scaling experiment costs nothing: 50k / 307k / full
are the same command with a different --corpus_size, each one a properly
fingerprinted artifact. Re-chunking would change the hash and invalidate
every embedding and cache built against it.


OUTPUT
------
  <out_dir>/nq_chunks_full.pt        Pass 1 cache (reused automatically)
  <out_dir>/nq_corpus.pt
      {"corpus": List[str], "titles": List[str], "meta": {...}}
  <out_dir>/nq_qa_train.pt
  <out_dir>/nq_qa_validation.pt
      {"records": [{question, answers, answerable, gold_locatable,
                    gold_ids, neg_ids, article_key, split, orig_idx}], "meta"}
  <out_dir>/nq_corpus_meta.json

Gold passages are stored as INTEGER doc_ids. Every downstream recall
computation is an integer set comparison -- no normalization, no substring
heuristics, no drift between scripts.


USAGE
-----
    # inspect dataset structure, build nothing
    python build_nq_corpus.py --out_dir /tmp/x --inspect_only

    # smoke test on 2000 articles (~1 min) before committing
    python build_nq_corpus.py --out_dir /tmp/nq_smoke --max_articles 2000

    # real build: headline 500K corpus
    python build_nq_corpus.py \
        --out_dir /home/mansoor/RAG/Natural_Questions/corpus \
        --corpus_size 500000

    # scaling sweep -- reuses Pass 1, no re-chunking
    python build_nq_corpus.py --out_dir <dir>_50k  --corpus_size 50000 \
        --chunks_cache <dir>/nq_chunks_full.pt
    python build_nq_corpus.py --out_dir <dir>_full --corpus_size 0 \
        --chunks_cache <dir>/nq_chunks_full.pt
"""

import os
import re
import sys
import json
import time
import random
import hashlib
import argparse
import unicodedata
from typing import List, Dict, Any, Tuple, Optional

import torch
from tqdm import tqdm
from datasets import load_dataset


DATASET = "lighteval/natural_questions_helm"
REQUIRED_FIELDS = ["id", "title", "document", "question",
                   "long_answers", "short_answers"]


# ----------------------------------------------------------------------------
# Structure validation -- run BEFORE any work, fail loudly on surprises
# ----------------------------------------------------------------------------
def validate_structure(ex: Dict[str, Any], split: str) -> None:
    """
    Assert the example has the exact shape this script depends on.

    We do not silently adapt to a different schema. If HuggingFace changes
    the layout we want a hard failure here, not wrong embeddings discovered
    three modules later.
    """
    missing = [f for f in REQUIRED_FIELDS if f not in ex]
    if missing:
        raise ValueError(
            f"[{split}] Missing required field(s): {missing}\n"
            f"Fields present: {list(ex.keys())}"
        )
    for f in ("title", "document", "question"):
        if not isinstance(ex[f], str):
            raise ValueError(
                f"[{split}] Expected ex['{f}'] to be str, got {type(ex[f])}"
            )
    for f in ("long_answers", "short_answers"):
        if not isinstance(ex[f], (list, tuple)):
            raise ValueError(
                f"[{split}] Expected ex['{f}'] to be a list, got {type(ex[f])}"
            )


def print_inspection(ex: Dict[str, Any], split: str) -> None:
    """Show the caller exactly what the data looks like."""
    print(f"\n--- structure sample: {split}[0] ---")
    print(f"  fields          : {list(ex.keys())}")
    print(f"  title           : {ex['title']!r}")
    print(f"  question        : {ex['question']!r}")
    print(f"  document chars  : {len(ex['document']):,}")
    print(f"  document head   : {ex['document'][:120]!r}")
    print(f"  n long_answers  : {len(ex['long_answers'])}")
    if ex["long_answers"]:
        print(f"  long_answers[0] : {ex['long_answers'][0][:120]!r}")
    print(f"  short_answers   : {[a[:60] for a in ex['short_answers'][:3]]}")


# ----------------------------------------------------------------------------
# Canonicalization -- the only reliable bridge between `document` and
# `long_answers`, which come from different text-extraction passes.
# ----------------------------------------------------------------------------
_CITE_RE = re.compile(r"\[\d+\]")


def canon_map(s: str) -> Tuple[str, List[int]]:
    """
    Canonicalize a string and return a map from canonical index -> raw index.

    Canonical form: NFKD, combining marks stripped (Rontgen == Röntgen),
    lowercased, [12] citation markers removed, non-alphanumerics dropped.
    That last step is what absorbs the tokenizer-spacing difference between
    the two extraction passes -- `document` writes " ( season 8 ) " and
    " ," while `long_answers` writes "(season 8)" and ",".

    The index map is what lets us convert a canonical match position back
    into a raw character span, which is what chunk boundaries live in.
    """
    out: List[str] = []
    idx: List[int] = []
    i, n = 0, len(s)
    while i < n:
        if s[i] == "[":
            m = _CITE_RE.match(s, i)
            if m:
                i = m.end()
                continue
        d = unicodedata.normalize("NFKD", s[i])
        d = "".join(c for c in d if not unicodedata.combining(c)).lower()
        for c in d:
            if c.isalnum():
                out.append(c)
                idx.append(i)
        i += 1
    return "".join(out), idx


def locate_gold_span(
    doc_canon: str,
    doc_map: List[int],
    gold_text: str,
    prefix_chars: int,
) -> Optional[Tuple[int, int]]:
    """
    Find a gold passage inside its article, returning a RAW character span.

    Two attempts, in order:
      1. Full canonical match.
      2. First `prefix_chars` canonical chars only. This catches golds whose
         opening matches but which diverge later -- typically because the
         gold's tail is a table that the two passes flattened differently.
         Measured contribution: +5.2 points on top of 84.9%.

    Returns None for the ~10% that are wholly table/infobox derived. Those
    are counted and reported, never silently dropped.
    """
    if not gold_text:
        return None
    g_canon, _ = canon_map(gold_text)
    if not g_canon:
        return None

    p = doc_canon.find(g_canon)
    n_match = len(g_canon)
    if p < 0 and len(g_canon) > prefix_chars:
        p = doc_canon.find(g_canon[:prefix_chars])
        n_match = prefix_chars
    if p < 0:
        return None

    end_c = min(p + n_match - 1, len(doc_map) - 1)
    if end_c < p:
        return None
    return doc_map[p], doc_map[end_c] + 1


# ----------------------------------------------------------------------------
# Chunking -- structural packing, not fixed offsets
# ----------------------------------------------------------------------------
_SPLIT_RE = re.compile(r"\n+|(?<=[.!?])\s{2,}")


def structural_units(doc: str) -> List[Tuple[int, int]]:
    """
    Split an article on its own structural boundaries.

    Newlines and multi-space breaks after sentence punctuation. These are
    the boundaries the article's paragraphs already sit on, which is why
    packing them keeps golds intact: measured p50 202 chars per unit,
    ~56 units per article, against a p50 gold of 527 raw chars.
    """
    parts: List[Tuple[int, int]] = []
    pos = 0
    for m in _SPLIT_RE.finditer(doc):
        if m.start() > pos:
            parts.append((pos, m.start()))
        pos = m.end()
    if pos < len(doc):
        parts.append((pos, len(doc)))
    return parts or [(0, len(doc))]


def pack_units(units: List[Tuple[int, int]], budget: int) -> List[Tuple[int, int]]:
    """
    Greedily merge consecutive structural units up to a character budget.

    A unit longer than the budget becomes its own chunk rather than being
    split -- splitting it would reintroduce exactly the mid-gold cut that
    structural packing exists to avoid.
    """
    out: List[Tuple[int, int]] = []
    cs: Optional[int] = None
    ce: Optional[int] = None
    for a, b in units:
        if cs is None:
            cs, ce = a, b
        elif b - cs <= budget:
            ce = b
        else:
            out.append((cs, ce))
            cs, ce = a, b
    if cs is not None:
        out.append((cs, ce))
    return out


def split_oversized(doc: str, a: int, b: int, cap: int, overlap: int) -> List[Tuple[int, int]]:
    """
    Fallback for structural units larger than `cap`.

    Structural packing never splits a unit, which is right when units are
    paragraph-sized. Some articles have no internal structure at all, so the
    entire article arrives as one unit -- measured max 94,234 chars. The
    encoder reads ~1,000 chars (max_length=256), so everything past that is
    invisible: 5.2% of chunks held 22% of all text, and 10.6% of GOLD chunks
    exceeded 2,000 chars.

    Split on whitespace near the cap, with overlap so a gold sitting on a
    split boundary still lands whole in one piece.
    """
    if b - a <= cap:
        return [(a, b)]
    stride = max(1, cap - overlap)
    out, i = [], a
    while i < b:
        j = min(i + cap, b)
        if j < b:                              # back off to a whitespace boundary
            k = doc.rfind(" ", i + stride, j)
            if k > i:
                j = k
        out.append((i, j))
        if j >= b:
            break
        i = max(i + stride, j - overlap)
    return out


def chunk_article(doc: str, budget: int, min_chars: int,
                  cap: int, overlap: int) -> List[Tuple[int, int]]:
    """Chunk spans for one article, dropping fragments below min_chars."""
    spans = []
    for a, b in pack_units(structural_units(doc), budget):
        spans.extend(split_oversized(doc, a, b, cap, overlap))
    kept = [(a, b) for a, b in spans if len(doc[a:b].strip()) >= min_chars]
    return kept or ([(spans[0][0], spans[0][1])] if spans else [])


def sha1_16(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


# ----------------------------------------------------------------------------
# Pass 1 -- every chunk of every article, deduplicated, deterministic
# ----------------------------------------------------------------------------
def build_all_chunks(
    splits: Dict[str, Any],
    budget: int,
    min_chars: int,
    max_articles: int,
    cap: int,
    overlap: int,
) -> Dict[str, Any]:
    """
    Scan all splits, deduplicate articles by text hash, chunk each once.

    Articles are deduplicated by TEXT, not title: NQ has 179,870
    title->different-text conflicts, so the title is not an identity.

    Chunks are then deduplicated globally by chunk text hash, and doc_ids
    are assigned by iterating articles in sorted (title, article_key) order
    and chunks in document order. That makes doc_id a pure function of the
    data -- reproducible across machines, processes and PYTHONHASHSEED,
    which matters because FAISS row order and top-k tie-breaking depend on it.
    """
    # --- deduplicate articles ------------------------------------------
    articles: Dict[str, Tuple[str, str]] = {}   # key -> (title, text)
    examples: List[Dict[str, Any]] = []

    for split_name, ds in splits.items():
        for i, ex in enumerate(tqdm(ds, desc=f"scanning {split_name}", unit="ex")):
            doc = ex["document"]
            if not doc or not ex["question"]:
                continue
            key = sha1_16(doc)
            if key not in articles:
                if max_articles and len(articles) >= max_articles:
                    continue
                articles[key] = (ex["title"], doc)
            examples.append({
                "split": split_name,
                "orig_idx": i,
                "question": ex["question"],
                "answers": list(ex["short_answers"]),
                "long_answers": list(ex["long_answers"]),
                "article_key": key,
                "title": ex["title"],
            })

    if max_articles:
        keep = set(articles.keys())
        examples = [e for e in examples if e["article_key"] in keep]

    print(f"\n  unique articles       : {len(articles):,}")
    print(f"  examples              : {len(examples):,}")

    # --- chunk in deterministic order ----------------------------------
    order = sorted(articles.keys(), key=lambda k: (articles[k][0], k))

    chunk_texts: List[str] = []
    chunk_titles: List[str] = []
    hash_to_id: Dict[str, int] = {}
    article_chunks: Dict[str, List[Tuple[int, int, int]]] = {}  # key -> [(doc_id, s, e)]

    n_dup = 0
    for key in tqdm(order, desc="chunking", unit="art"):
        title, doc = articles[key]
        spans = chunk_article(doc, budget, min_chars, cap, overlap)
        entries: List[Tuple[int, int, int]] = []
        for a, b in spans:
            text = doc[a:b].strip()
            if not text:
                continue
            h = sha1_16(text)
            did = hash_to_id.get(h)
            if did is None:
                did = len(chunk_texts)
                hash_to_id[h] = did
                chunk_texts.append(text)
                chunk_titles.append(title)
            else:
                n_dup += 1
            entries.append((did, a, b))
        article_chunks[key] = entries

    print(f"  unique chunks         : {len(chunk_texts):,}")
    print(f"  duplicate chunks      : {n_dup:,}")

    # --- align golds ----------------------------------------------------
    n_answerable = n_located = n_unlocated = 0
    gold_span_hist: Dict[int, int] = {}

    for ex in tqdm(examples, desc="aligning gold", unit="ex"):
        ex["gold_ids"] = []
        ex["gold_locatable"] = False
        ex["answerable"] = bool(ex["answers"])
        if not ex["long_answers"]:
            continue
        n_answerable += 1

        title, doc = articles[ex["article_key"]]
        d_canon, d_map = canon_map(doc)

        gold_ids: List[int] = []
        seen = set()
        for gtext in ex["long_answers"]:
            span = locate_gold_span(d_canon, d_map, gtext, prefix_chars=200)
            if span is None:
                continue
            gs, ge = span
            for did, a, b in article_chunks[ex["article_key"]]:
                if a < ge and b > gs and did not in seen:
                    seen.add(did)
                    gold_ids.append(did)

        if gold_ids:
            ex["gold_ids"] = gold_ids
            ex["gold_locatable"] = True
            n_located += 1
            gold_span_hist[len(gold_ids)] = gold_span_hist.get(len(gold_ids), 0) + 1
        else:
            n_unlocated += 1

    print(f"\n  answerable examples   : {n_answerable:,}")
    print(f"  gold located          : {n_located:,} "
          f"({100.0 * n_located / max(1, n_answerable):.1f}%)")
    print(f"  gold NOT located      : {n_unlocated:,} "
          f"({100.0 * n_unlocated / max(1, n_answerable):.1f}%)  "
          f"[flattened tables/infoboxes]")
    print(f"  gold-chunk-count hist : {dict(sorted(gold_span_hist.items())[:6])}")

    h = hashlib.sha256()
    for t in chunk_texts:
        h.update(sha1_16(t).encode("utf-8"))
    full_hash = h.hexdigest()[:16]

    return {
        "chunk_texts": chunk_texts,
        "chunk_titles": chunk_titles,
        "article_chunks": article_chunks,
        "examples": examples,
        "diagnostics": {
            "n_articles": len(articles),
            "n_chunks": len(chunk_texts),
            "n_duplicate_chunks": n_dup,
            "n_examples": len(examples),
            "n_answerable": n_answerable,
            "n_gold_located": n_located,
            "n_gold_unlocated": n_unlocated,
            "gold_chunk_count_histogram": dict(sorted(gold_span_hist.items())),
            "chunk_budget": budget,
            "min_chunk_chars": min_chars,
            "max_chunk_chars": cap,
            "split_overlap": overlap,
        },
        "full_corpus_hash": full_hash,
    }


# ----------------------------------------------------------------------------
# Pass 2 -- deterministic subsample at --corpus_size
# ----------------------------------------------------------------------------
def subsample_corpus(
    pass1: Dict[str, Any],
    corpus_size: int,
    seed: int,
) -> Tuple[List[str], List[str], Dict[int, int], Dict[str, Any]]:
    """
    Select a corpus of `corpus_size` chunks, retaining EVERY gold chunk.

    Golds are never sampled out. Dropping a gold would leave its example
    with no findable answer, which does not lower that example's recall --
    it makes the metric meaningless for it.

    Returns (corpus, titles, old_id -> new_id, diagnostics).
    """
    n_total = len(pass1["chunk_texts"])

    gold_ids = set()
    for ex in pass1["examples"]:
        gold_ids.update(ex["gold_ids"])

    if corpus_size <= 0 or corpus_size >= n_total:
        selected = list(range(n_total))
        n_filler = n_total - len(gold_ids)
    else:
        if len(gold_ids) > corpus_size:
            raise ValueError(
                f"--corpus_size {corpus_size:,} is smaller than the number of "
                f"gold chunks ({len(gold_ids):,}). Every gold must be retained, "
                f"so raise --corpus_size to at least {len(gold_ids):,}."
            )
        rest = [i for i in range(n_total) if i not in gold_ids]
        n_filler = corpus_size - len(gold_ids)
        rng = random.Random(seed)
        rng.shuffle(rest)
        selected = sorted(gold_ids) + sorted(rest[:n_filler])
        selected.sort()

    old2new = {old: new for new, old in enumerate(selected)}
    corpus = [pass1["chunk_texts"][i] for i in selected]
    titles = [pass1["chunk_titles"][i] for i in selected]

    diagnostics = {
        "n_chunks_available": n_total,
        "n_chunks_selected": len(selected),
        "n_gold_chunks": len(gold_ids),
        "n_filler_chunks": n_filler,
        "gold_fraction_of_corpus": round(len(gold_ids) / max(1, len(selected)), 5),
        "subsample_seed": seed,
    }
    return corpus, titles, old2new, diagnostics


def build_qa_records(
    pass1: Dict[str, Any],
    old2new: Dict[int, int],
    max_negs: int,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    """
    Convert examples into QA records with remapped integer doc_ids.

    neg_ids are the example's own article's non-gold chunks. NQ has no
    distractor field, so this is the closest analogue to HotpotQA's
    distractor paragraphs: topically adjacent by construction, and far
    stronger negatives than uniform corpus samples.

    Chunks not selected into this corpus are filtered out of both gold_ids
    and neg_ids -- golds never are, by construction of the subsample.
    """
    by_split: Dict[str, List[Dict[str, Any]]] = {}
    diags: Dict[str, Any] = {}

    for ex in tqdm(pass1["examples"], desc="records", unit="ex"):
        gold = [old2new[g] for g in ex["gold_ids"] if g in old2new]
        gold_set = set(ex["gold_ids"])

        negs: List[int] = []
        for did, _a, _b in pass1["article_chunks"].get(ex["article_key"], []):
            if did in gold_set:
                continue
            new = old2new.get(did)
            if new is not None and new not in negs:
                negs.append(new)
            if len(negs) >= max_negs:
                break

        rec = {
            "question": ex["question"],
            "answers": ex["answers"],
            "answerable": ex["answerable"],
            "gold_locatable": ex["gold_locatable"] and bool(gold),
            "gold_ids": gold,
            "neg_ids": negs,
            "article_key": ex["article_key"],
            "title": ex["title"],
            "split": ex["split"],
            "orig_idx": ex["orig_idx"],
        }
        by_split.setdefault(ex["split"], []).append(rec)

    for split_name, recs in by_split.items():
        gold_hist: Dict[int, int] = {}
        neg_hist: Dict[int, int] = {}
        n_ans = n_loc = n_train_ready = 0
        for r in recs:
            gold_hist[len(r["gold_ids"])] = gold_hist.get(len(r["gold_ids"]), 0) + 1
            nb = min(len(r["neg_ids"]), 10)
            neg_hist[nb] = neg_hist.get(nb, 0) + 1
            n_ans += int(r["answerable"])
            n_loc += int(r["gold_locatable"])
            n_train_ready += int(r["answerable"] and r["gold_locatable"])
        diags[split_name] = {
            "n_examples": len(recs),
            "n_answerable": n_ans,
            "n_gold_locatable": n_loc,
            "n_usable_for_generation": n_train_ready,
            "gold_count_histogram": dict(sorted(gold_hist.items())),
            "neg_count_histogram_capped10": dict(sorted(neg_hist.items())),
        }

    return by_split, diags


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Build a fixed pooled retrieval corpus for Natural Questions."
    )
    p.add_argument("--out_dir", type=str, required=True,
                   help="Directory to write nq_corpus.pt and nq_qa_{split}.pt")
    p.add_argument("--dataset", type=str, default=DATASET,
                   help="HuggingFace dataset path")
    p.add_argument("--splits", type=str, nargs="+",
                   default=["train", "validation"],
                   help="Splits to pool into the corpus")
    p.add_argument("--chunk_budget", type=int, default=1000,
                   help="Character budget per chunk (structural packing). "
                        "Measured: 92.9%% of golds land whole at 1000, with "
                        "gold_ids p50/p90 = [1,1].")
    p.add_argument("--min_chunk_chars", type=int, default=50,
                   help="Drop chunk fragments shorter than this.")
    p.add_argument("--max_chunk_chars", type=int, default=1500,
                   help="Hard cap. Structural units above this are split with "
                        "overlap. Measured: 10.6%% of golds exceeded 2000 chars "
                        "and were unreachable at max_length=256.")
    p.add_argument("--split_overlap", type=int, default=200,
                   help="Overlap when splitting an oversized unit, so a gold "
                        "on a split boundary still lands whole in one piece.")
    p.add_argument("--corpus_size", type=int, default=500000,
                   help="Chunks in the final corpus. 0 = use all. Every gold "
                        "chunk is retained regardless.")
    p.add_argument("--subsample_seed", type=int, default=42,
                   help="Seed for filler-chunk selection. The corpus is data, "
                        "not a model artifact -- this is NOT the training seed.")
    p.add_argument("--max_negs", type=int, default=16,
                   help="Max same-article non-gold chunks stored per example.")
    p.add_argument("--chunks_cache", type=str, default=None,
                   help="Path to an existing nq_chunks_full.pt (Pass 1). "
                        "Defaults to <out_dir>/nq_chunks_full.pt.")
    p.add_argument("--max_articles", type=int, default=0,
                   help="Smoke test: chunk only the first N unique articles.")
    p.add_argument("--inspect_only", action="store_true",
                   help="Print dataset structure and exit without building.")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing output files.")
    return p.parse_args()


def main():
    args = parse_args()
    t_start = time.time()

    print("=" * 72)
    print(" Natural Questions corpus builder")
    print("=" * 72)
    print(f"  dataset       : {args.dataset}")
    print(f"  splits        : {args.splits}")
    print(f"  chunk budget  : {args.chunk_budget} chars (structural packing)")
    print(f"  corpus size   : {args.corpus_size or 'ALL'}")
    print(f"  out_dir       : {args.out_dir}")
    if args.max_articles:
        print(f"  SMOKE TEST    : first {args.max_articles:,} articles only")
    print("=" * 72)

    # --- Validate all preconditions BEFORE any work ------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    corpus_path = os.path.join(args.out_dir, "nq_corpus.pt")
    if os.path.exists(corpus_path) and not args.overwrite:
        raise FileExistsError(
            f"{corpus_path} already exists. Pass --overwrite to replace it.\n"
            "(Refusing silently to overwrite a corpus that embeddings may "
            "already have been built against.)"
        )
    probe = os.path.join(args.out_dir, ".write_probe")
    with open(probe, "w") as f:
        f.write("ok")
    os.remove(probe)

    cache_path = args.chunks_cache or os.path.join(args.out_dir, "nq_chunks_full.pt")

    # --- Load and validate splits ------------------------------------------
    splits = {}
    for name in args.splits:
        print(f"\nLoading split '{name}' ...")
        splits[name] = load_dataset(args.dataset, split=name)
        print(f"  {name}: {len(splits[name]):,} examples")

    print("\nValidating structure ...")
    for name, ds in splits.items():
        validate_structure(ds[0], name)
        print_inspection(ds[0], name)
    print("\n  structure OK")

    if args.inspect_only:
        print("\n--inspect_only set; exiting without building.")
        return

    # --- Pass 1 ------------------------------------------------------------
    if os.path.exists(cache_path):
        print(f"\n[1/3] Loading Pass 1 cache: {cache_path}")
        pass1 = torch.load(cache_path, map_location="cpu", weights_only=False)
        d = pass1["diagnostics"]
        if (d["chunk_budget"] != args.chunk_budget
                or d.get("max_chunk_chars") != args.max_chunk_chars):
            raise ValueError(
                f"Cache was built with chunk_budget={d['chunk_budget']}, "
                f"max_chunk_chars={d.get('max_chunk_chars')} but "
                f"--chunk_budget={args.chunk_budget}, "
                f"--max_chunk_chars={args.max_chunk_chars} was requested. "
                f"Chunking changes every doc_id; delete the cache to rebuild."
            )
        print(f"  chunks : {d['n_chunks']:,}   full hash: {pass1['full_corpus_hash']}")
    else:
        print("\n[1/3] Pass 1: chunking all articles (run once) ...")
        pass1 = build_all_chunks(
            splits, args.chunk_budget, args.min_chunk_chars, args.max_articles,
            args.max_chunk_chars, args.split_overlap
        )
        print(f"\n  saving Pass 1 cache -> {cache_path}")
        torch.save(pass1, cache_path)
        print(f"  full corpus hash    : {pass1['full_corpus_hash']}")

    lengths = [len(c) for c in pass1["chunk_texts"]]
    lengths.sort()
    print(f"  chunk chars min/med/max: "
          f"{lengths[0]} / {lengths[len(lengths) // 2]} / {lengths[-1]}")

    # --- Pass 2 ------------------------------------------------------------
    print("\n[2/3] Pass 2: subsampling corpus ...")
    corpus, titles, old2new, sub_diag = subsample_corpus(
        pass1, args.corpus_size, args.subsample_seed
    )
    print(f"  chunks available      : {sub_diag['n_chunks_available']:,}")
    print(f"  chunks selected       : {sub_diag['n_chunks_selected']:,}")
    print(f"  gold chunks (all kept): {sub_diag['n_gold_chunks']:,}")
    print(f"  gold fraction         : {sub_diag['gold_fraction_of_corpus']:.4f}")
    if sub_diag["gold_fraction_of_corpus"] > 0.25:
        print("  WARNING: golds exceed 25% of the corpus. Retrieval will be "
              "artificially easy. Raise --corpus_size before trusting recall.")

    h = hashlib.sha256()
    for t in corpus:
        h.update(sha1_16(t).encode("utf-8"))
    corpus_hash = h.hexdigest()[:16]
    print(f"  corpus fingerprint    : {corpus_hash}")

    print("\n  building QA records with integer gold ids ...")
    qa_records, qa_diags = build_qa_records(pass1, old2new, args.max_negs)
    for name, d in qa_diags.items():
        print(f"\n  [{name}]")
        print(f"    examples                 : {d['n_examples']:,}")
        print(f"    answerable               : {d['n_answerable']:,}")
        print(f"    gold locatable           : {d['n_gold_locatable']:,}")
        print(f"    usable for generation    : {d['n_usable_for_generation']:,}")
        gh = dict(list(d["gold_count_histogram"].items())[:6])
        print(f"    gold-count histogram     : {gh}")
        nh = dict(list(d["neg_count_histogram_capped10"].items())[:6])
        print(f"    neg-count hist (cap 10)  : {nh}")

    # --- Save --------------------------------------------------------------
    print("\n[3/3] Saving ...")
    meta = {
        "dataset": args.dataset,
        "splits": args.splits,
        "chunking": "structural_packing",
        "chunk_budget": args.chunk_budget,
        "min_chunk_chars": args.min_chunk_chars,
        "n_docs": len(corpus),
        "corpus_hash": corpus_hash,
        "full_corpus_hash": pass1["full_corpus_hash"],
        "corpus_size_requested": args.corpus_size,
        "subsample_seed": args.subsample_seed,
        "max_negs": args.max_negs,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "builder_script": os.path.basename(__file__),
        "python": sys.version.split()[0],
        "pass1_diagnostics": pass1["diagnostics"],
        "subsample_diagnostics": sub_diag,
        "qa_diagnostics": qa_diags,
        "max_chunk_chars": args.max_chunk_chars,
        "split_overlap": args.split_overlap,
    }

    torch.save({"corpus": corpus, "titles": titles, "meta": meta}, corpus_path)
    print(f"  corpus  -> {corpus_path}")

    for name, recs in qa_records.items():
        qa_path = os.path.join(args.out_dir, f"nq_qa_{name}.pt")
        torch.save({"records": recs, "meta": meta}, qa_path)
        print(f"  qa      -> {qa_path}  ({len(recs):,} records)")

    meta_path = os.path.join(args.out_dir, "nq_corpus_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"  meta    -> {meta_path}")

    print(f"\nDone in {time.time() - t_start:.1f}s")
    print(f"Corpus fingerprint {corpus_hash} -- quote this in downstream artifacts.")


if __name__ == "__main__":
    main()
