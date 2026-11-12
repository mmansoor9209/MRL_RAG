#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_nq_crag_baseline.py

CRAG baseline (Yan et al., 2024 -- Corrective Retrieval Augmented Generation)
on our NQ corpus, corpus-only.

CRAG COMPONENTS, AS IMPLEMENTED
-------------------------------
1. Retrieval evaluator: scores each retrieved document for relevance to the
   question.
2. Confidence-triggered action, from the per-query evaluator scores:
     Correct   : at least one doc scores above UPPER
                 -> use refined internal knowledge
     Incorrect : every doc scores below LOWER
                 -> CRAG discards the retrieval and uses web search.
                    A fixed corpus allows no external search, so the
                    generator answers from the question alone.
     Ambiguous : otherwise
                 -> CRAG combines refined internal knowledge with web
                    results; here refined internal knowledge only.
3. Knowledge refinement (decompose-then-recompose): split retrieved docs
   into sentence-level strips, score every strip with the evaluator, keep
   the highest-scoring strips, and recompose them in original order.

DIFFERENCES FROM THE PAPER (state in the paper)
-----------------------------------------------
* Web search disabled (fixed-corpus evaluation), handled as above.
* Evaluator = cross-encoder/ms-marco-MiniLM-L-6-v2 (CRAG fine-tunes a
  T5-large evaluator). UPPER / LOWER are calibrated on NQ TRAINING queries
  against gold-chunk labels, never on the eval set:
     UPPER (per doc)   = smallest score with precision(gold | score >= UPPER) >= 0.5
     LOWER (per query) = largest best-doc score with
                         P(top-k contains a gold | best score < LOWER) <= 0.10
  Calibration makes the thresholds independent of the evaluator's scale.
* Retriever: zero-shot Contriever-msmarco top-k. Generator: our Llama-3 8B
  + the shared LoRA adapter, like every other generation baseline. The
  refined context is capped at 2,700 chars (= 3 x 900, the same context
  budget as our method's top-3 chunks).

USAGE
-----
    CUDA_VISIBLE_DEVICES=3 python eval_nq_crag_baseline.py \
        --corpus_dir /home/mansoor/RAG/Natural_Questions/corpus \
        --out_dir    /home/mansoor/RAG/Natural_Questions/baselines \
        --emb_cache  /home/mansoor/RAG/Natural_Questions/baselines/contriever_zs_embs.pt \
        --lora_path  .../joint_contriever/aligned_seed42/generator_lora \
        --n_eval 100
"""

import os
import re
import json
import time
import random
import argparse
from collections import Counter
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from mcf_nq_module4_joint import build_prompt, extract_answer, em_f1_multi
from nq_zeroshot_retrieval import get_corpus_embs, retrieve

CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def load_usable(corpus_dir: str, split: str, corpus_hash: str):
    qa = torch.load(os.path.join(corpus_dir, f"nq_qa_{split}.pt"),
                    map_location="cpu", weights_only=False)
    if qa["meta"]["corpus_hash"] != corpus_hash:
        raise ValueError(f"corpus_hash mismatch in nq_qa_{split}.pt")
    return [r for r in qa["records"]
            if r.get("gold_locatable") and r["gold_ids"] and r.get("answers")]


def score_docs(ce, questions: List[str], ids: np.ndarray, corpus: List[str],
               max_chars: int, batch_size: int) -> np.ndarray:
    n, k = ids.shape
    pairs = [[questions[i], corpus[int(ids[i, j])][:max_chars]]
             for i in range(n) for j in range(k)]
    return np.asarray(ce.predict(pairs, batch_size=batch_size,
                                 show_progress_bar=True), dtype=np.float32).reshape(n, k)


def calibrate(scores: np.ndarray, labels: np.ndarray,
              upper_precision: float, lower_gold_rate: float) -> Tuple[float, float]:
    """
    Data-driven UPPER / LOWER from training scores (n, k) and gold labels (n, k).

    UPPER is per-document, because 'Correct' fires when ANY single doc is
    confidently relevant:
        smallest t with precision(doc is gold | doc score >= t) >= upper_precision
    LOWER is per-query on the best doc score, because 'Incorrect' fires when
    EVERY doc scores low, i.e. when the max score is below LOWER:
        largest t with P(top-k contains a gold | max score < t) <= lower_gold_rate
    """
    s, y = scores.ravel(), labels.ravel()
    grid = np.unique(np.quantile(s, np.linspace(0.0, 1.0, 401)))
    upper = float(grid[-1])
    for t in grid:
        m = s >= t
        if m.sum() >= 20 and y[m].mean() >= upper_precision:
            upper = float(t)
            break

    q_max = scores.max(axis=1)
    q_hit = labels.max(axis=1)
    q_grid = np.unique(np.quantile(q_max, np.linspace(0.0, 1.0, 401)))
    lower = float(q_grid[0])
    for t in q_grid:
        m = q_max < t
        if m.sum() >= 20 and q_hit[m].mean() <= lower_gold_rate:
            lower = float(t)
    return upper, min(lower, upper)


def refine(ce, question: str, docs: List[str], strip_sents: int, n_strips: int,
           min_score: float, max_ctx_chars: int, batch_size: int) -> str:
    """Decompose docs into strips, keep the best, recompose in original order."""
    strips = []                                  # (doc_rank, position, text)
    for di, d in enumerate(docs):
        sents = [x for x in _SENT_SPLIT.split(d) if x.strip()]
        for si in range(0, len(sents), strip_sents):
            strips.append((di, si, " ".join(sents[si:si + strip_sents])))
    if not strips:
        return ""
    sc = ce.predict([[question, s[2]] for s in strips], batch_size=batch_size,
                    show_progress_bar=False)
    order = np.argsort(-np.asarray(sc))
    keep = [j for j in order if sc[j] >= min_score][:n_strips]
    if not keep:
        keep = list(order[:2])                   # never return an empty context
    keep.sort(key=lambda j: (strips[j][0], strips[j][1]))
    return " ".join(strips[j][2] for j in keep)[:max_ctx_chars]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--emb_cache", default=None,
                   help="Zero-shot Contriever corpus embedding cache (shared with Self-RAG).")
    p.add_argument("--lora_path", required=True,
                   help="Same adapter as every other generation baseline.")
    p.add_argument("--hf_model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--top_k", type=int, default=5,
                   help="Docs retrieved and evaluated per query (CRAG uses top-5).")
    p.add_argument("--n_calib", type=int, default=2000,
                   help="NQ TRAIN queries used to calibrate UPPER/LOWER.")
    p.add_argument("--upper_precision", type=float, default=0.5)
    p.add_argument("--lower_gold_rate", type=float, default=0.10,
                   help="Max rate of queries with a gold in top-k among those called Incorrect.")
    p.add_argument("--strip_sents", type=int, default=2)
    p.add_argument("--n_strips", type=int, default=6)
    p.add_argument("--max_ctx_chars", type=int, default=2700,
                   help="3 x 900: same context budget as our top-3 chunks.")
    p.add_argument("--ce_batch_size", type=int, default=256)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--n_eval", type=int, default=0, help="0 = all 3,950")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"crag_k{args.top_k}" + (f"_n{args.n_eval}" if args.n_eval else "")
    out_path = os.path.join(args.out_dir, f"baseline_{tag}_nq.json")
    if os.path.exists(out_path) and not args.overwrite:
        raise FileExistsError(f"{out_path} exists. Pass --overwrite.")

    # ---- data --------------------------------------------------------------
    cobj = torch.load(os.path.join(args.corpus_dir, "nq_corpus.pt"),
                      map_location="cpu", weights_only=False)
    corpus = cobj["corpus"]
    corpus_hash = cobj["meta"]["corpus_hash"]
    recs = load_usable(args.corpus_dir, "validation", corpus_hash)
    if args.n_eval:
        recs = recs[:args.n_eval]
    calib = load_usable(args.corpus_dir, "train", corpus_hash)[:args.n_calib]

    print("=" * 72)
    print(" CRAG BASELINE (Yan et al., 2024) -- Natural Questions, corpus-only")
    print("=" * 72)
    print(f"  corpus     : {len(corpus):,} chunks  [hash {corpus_hash}]")
    print(f"  eval       : {len(recs):,} queries   calibration: {len(calib):,} TRAIN queries")
    print(f"  retriever  : zero-shot Contriever-msmarco, top-{args.top_k}")
    print(f"  evaluator  : {CE_MODEL}")
    print(f"  generator  : {args.hf_model} + {os.path.basename(args.lora_path.rstrip('/'))}")
    print("=" * 72)

    # ---- retrieval ---------------------------------------------------------
    print("\n[1/4] retrieval")
    D = get_corpus_embs(corpus, corpus_hash, args.emb_cache, device)
    ev_q = [r["question"] for r in recs]
    ca_q = [r["question"] for r in calib]
    ev_ids, _ = retrieve(ev_q, D, args.top_k, device)
    ca_ids, _ = retrieve(ca_q, D, args.top_k, device)
    del D

    # ---- evaluator + calibration ------------------------------------------
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(CE_MODEL, max_length=512, device=str(device))

    print("\n[2/4] calibrating thresholds on TRAIN queries")
    ca_sc = score_docs(ce, ca_q, ca_ids, corpus, 900, args.ce_batch_size)
    ca_lab = np.array([[int(int(ca_ids[i, j]) in set(calib[i]["gold_ids"]))
                        for j in range(args.top_k)] for i in range(len(calib))])
    upper, lower = calibrate(ca_sc, ca_lab, args.upper_precision, args.lower_gold_rate)
    print(f"  UPPER = {upper:.4f}   LOWER = {lower:.4f}   "
          f"(train gold rate in top-{args.top_k}: {ca_lab.mean():.4f})")

    print("\n[3/4] evaluating + refining eval queries")
    ev_sc = score_docs(ce, ev_q, ev_ids, corpus, 900, args.ce_batch_size)
    actions, contexts = [], []
    for i, r in enumerate(tqdm(recs, desc="  actions/refine", unit="q")):
        s = ev_sc[i]
        if s.max() > upper:
            act = "correct"
        elif s.max() < lower:
            act = "incorrect"
        else:
            act = "ambiguous"
        actions.append(act)
        if act == "incorrect":
            contexts.append(None)
        else:
            docs = [corpus[int(d)] for d in ev_ids[i]]
            contexts.append(refine(ce, r["question"], docs, args.strip_sents,
                                   args.n_strips, lower, args.max_ctx_chars,
                                   args.ce_batch_size))
    del ce
    torch.cuda.empty_cache()
    counts = Counter(actions)
    print("  actions: " + "  ".join(f"{a}={counts.get(a, 0)}"
                                    for a in ("correct", "ambiguous", "incorrect")))

    # ---- generation --------------------------------------------------------
    print("\n[4/4] generation")
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(args.hf_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    gen = AutoModelForCausalLM.from_pretrained(
        args.hf_model, torch_dtype=torch.bfloat16, device_map={"": 0})
    gen.config.use_cache = True
    gen = PeftModel.from_pretrained(gen, args.lora_path)
    gen.eval()

    ems, f1s, samples = [], [], []
    per_action = {a: [] for a in ("correct", "ambiguous", "incorrect")}
    for s in tqdm(range(0, len(recs), args.eval_batch_size), desc="  generating", unit="batch"):
        idx = list(range(s, min(s + args.eval_batch_size, len(recs))))
        prompts = []
        for i in idx:
            q = recs[i]["question"]
            if contexts[i] is None:                         # corpus-only 'incorrect'
                prompts.append(f"Question: {q}\nAnswer:")   # same as closed-book baseline
            else:
                prompts.append(build_prompt(q, [contexts[i]], args.max_ctx_chars))
        enc = tok(prompts, add_special_tokens=True, return_tensors="pt",
                  padding=True, truncation=True, max_length=args.max_len).to(device)
        with torch.no_grad():
            out = gen.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                               max_new_tokens=args.max_new_tokens, do_sample=False,
                               num_beams=1, pad_token_id=tok.pad_token_id)
        for b, i in enumerate(idx):
            txt = tok.decode(out[b, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            pred = extract_answer(txt)
            e, f = em_f1_multi(pred, recs[i]["answers"])
            ems.append(e); f1s.append(f)
            per_action[actions[i]].append(e)
            if len(samples) < 5:
                samples.append({"q": recs[i]["question"], "golds": recs[i]["answers"][:3],
                                "pred": pred, "action": actions[i]})

    em, f1 = float(np.mean(ems)), float(np.mean(f1s))
    print(f"\n  EM = {em * 100:.2f}    F1 = {f1 * 100:.2f}   (n={len(recs)})")
    for a, v in per_action.items():
        if v:
            print(f"    {a:<10} n={len(v):>5}  EM={np.mean(v) * 100:.2f}")
    for smp in samples:
        print(f"    Q: {smp['q'][:70]}\n       golds={smp['golds'][:2]}  "
              f"pred={smp['pred']!r}  action={smp['action']}")

    with open(out_path, "w") as f:
        json.dump({"baseline": "crag_corpus_only", "dataset": "natural_questions",
                   "corpus_hash": corpus_hash, "n_queries": len(recs),
                   "EM": em, "F1": f1, "upper": upper, "lower": lower,
                   "action_counts": dict(counts),
                   "em_by_action": {a: float(np.mean(v)) for a, v in per_action.items() if v},
                   "samples": samples,
                   "note": "Corpus-only CRAG: web search replaced by no-retrieval answer; "
                           "MS-MARCO cross-encoder evaluator with thresholds calibrated "
                           "on NQ train queries.",
                   "args": vars(args)}, f, indent=2)
    print(f"\n  -> {out_path}\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
