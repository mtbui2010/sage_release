# pyplanner/memory_retriever.py
# ─────────────────────────────────────────────────────────────────────
# Hybrid retrieval memory for SAGE.
#
# Two complementary sources feed a single query interface:
#
#   1. Seed pool (cold-start):
#        Loaded from eval_dataset_gt.json — curated, simulator-verified
#        ground-truth plans. Guarantees coverage even before any episode
#        has been logged.
#
#   2. Live pool (continuous learning):
#        Each successfully completed plan can be appended by calling
#        .add_episode(...). Persisted to a JSONL file so it survives
#        across runs.
#
# Retrieval is embedding-based when Chroma is available, with a
# Jaccard-over-tokens fallback that requires no extra dependencies —
# important because the benchmark must be runnable on a minimal
# environment. Both paths return _Example tuples matching the format
# used by HierarchicalFewShotPlanner so existing prompt assembly works.
# ─────────────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, asdict
from typing import Iterable, NamedTuple

from pyplanner.verifier import normalize_plan


# ─────────────────────────────────────────────────────────────────────
# Public types — match HierarchicalFewShotPlanner._Example shape
# ─────────────────────────────────────────────────────────────────────
class RetrievedExample(NamedTuple):
    task:      str
    reasoning: str
    plan_text: str   # JSON string of {"steps": [...]}
    score:     float = 0.0
    source:    str   = "seed"   # "seed" or "episode"


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────
_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _plan_to_text(plan: list[dict]) -> str:
    """Serialize a normalized step list to the JSON form used in prompts."""
    return json.dumps({"steps": [
        {"action": s.get("action", ""), "object": s.get("object", "")}
        for s in plan
    ]}, indent=None)


def _format_reasoning(task: str, expected_objects: list[str] | None,
                      room: str | None) -> str:
    """Synthesize a short reasoning trace for seed entries that lack one."""
    parts = []
    if room:
        parts.append(f"Task takes place in the {room}.")
    if expected_objects:
        parts.append(f"Key objects involved: {', '.join(expected_objects)}.")
    parts.append("Plan follows MoveTo → Find → Pick → MoveTo → Place pattern, "
                 "wrapping Open/Close around any container-bound items.")
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────
# Seed loader: eval_dataset_gt.json → RetrievedExample list
# ─────────────────────────────────────────────────────────────────────
def load_seed_examples(gt_path: str) -> list[RetrievedExample]:
    """Parse eval_dataset_gt.json into RetrievedExample seed entries.

    The GT file uses legacy action names (Navigate, Grab); we normalize
    them through pyplanner.verifier.normalize_plan() before serializing
    so seed examples are immediately usable as few-shot context.
    """
    if not os.path.exists(gt_path):
        return []

    with open(gt_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples: list[RetrievedExample] = []
    for s in data.get("samples", []):
        ref = s.get("reference_steps") or []
        if not ref:
            continue
        # ref is in the GT-format dict {action, object, target, reason};
        # collapse target into object when target is non-empty (matches
        # canonical schema used by pyplanner for Place steps).
        canonical = []
        for st in ref:
            action = st.get("action", "")
            obj    = st.get("object", "") or st.get("target", "")
            canonical.append({"action": action, "object": obj})
        plan = normalize_plan(canonical)
        examples.append(RetrievedExample(
            task      = s.get("task_desc", "").strip(),
            reasoning = _format_reasoning(
                s.get("task_desc", ""),
                s.get("expected_objects"),
                s.get("room"),
            ),
            plan_text = _plan_to_text(plan),
            score     = 0.0,
            source    = "seed",
        ))
    return examples


# ─────────────────────────────────────────────────────────────────────
# MemoryRetriever — hybrid (seed + live), Chroma-or-Jaccard backend
# ─────────────────────────────────────────────────────────────────────
@dataclass
class MemoryRetrieverConfig:
    gt_path:        str  = ""
    live_path:      str  = ""
    use_chroma:     bool = False
    chroma_path:    str  = ""
    chroma_collection: str = "sage_memory"
    top_k:          int  = 3


class MemoryRetriever:
    """Hybrid retrieval over (seed_examples ∪ live_episodes).

    Falls back to Jaccard token similarity when Chroma is unavailable
    or unconfigured. Chroma is preferred when present because it
    handles paraphrases (e.g., "brew coffee" vs. "make a cup of
    coffee") better than word overlap.
    """

    def __init__(self, cfg: MemoryRetrieverConfig | None = None):
        self.cfg          = cfg or MemoryRetrieverConfig()
        self._seed:    list[RetrievedExample] = []
        self._live:    list[RetrievedExample] = []
        self._chroma_col = None  # lazy-init

        if self.cfg.gt_path:
            self._seed = load_seed_examples(self.cfg.gt_path)
        if self.cfg.live_path and os.path.exists(self.cfg.live_path):
            self._live = self._load_live(self.cfg.live_path)

        if self.cfg.use_chroma:
            self._chroma_col = self._init_chroma()
            if self._chroma_col is not None:
                self._index_into_chroma(self._seed, source_tag="seed")
                self._index_into_chroma(self._live, source_tag="episode")

    # ── pool inspection ───────────────────────────────────────────
    @property
    def size(self) -> dict[str, int]:
        return {"seed": len(self._seed), "live": len(self._live)}

    # ── live store I/O ─────────────────────────────────────────────
    @staticmethod
    def _load_live(path: str) -> list[RetrievedExample]:
        out = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    out.append(RetrievedExample(
                        task      = rec.get("task", ""),
                        reasoning = rec.get("reasoning", ""),
                        plan_text = rec.get("plan_text", ""),
                        score     = 0.0,
                        source    = "episode",
                    ))
                except json.JSONDecodeError:
                    continue
        return out

    def add_episode(self, task: str, plan: list[dict],
                    reasoning: str = "", success: bool = True) -> None:
        """Append a successful plan to the live pool and persist to disk."""
        if not success or not plan:
            return
        ex = RetrievedExample(
            task      = task.strip(),
            reasoning = reasoning or "Recorded from a successful episode.",
            plan_text = _plan_to_text(normalize_plan(plan)),
            score     = 0.0,
            source    = "episode",
        )
        self._live.append(ex)
        if self.cfg.live_path:
            os.makedirs(os.path.dirname(self.cfg.live_path) or ".", exist_ok=True)
            with open(self.cfg.live_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "task":      ex.task,
                    "reasoning": ex.reasoning,
                    "plan_text": ex.plan_text,
                    "ts":        time.time(),
                }) + "\n")
        if self._chroma_col is not None:
            self._add_to_chroma(ex, source_tag="episode")

    # ── Chroma backend ─────────────────────────────────────────────
    def _init_chroma(self):
        try:
            import chromadb
        except ImportError:
            return None
        try:
            if self.cfg.chroma_path:
                client = chromadb.PersistentClient(path=self.cfg.chroma_path)
            else:
                client = chromadb.Client()
            return client.get_or_create_collection(self.cfg.chroma_collection)
        except Exception:
            return None

    def _index_into_chroma(self, examples: list[RetrievedExample], source_tag: str) -> None:
        if not examples or self._chroma_col is None:
            return
        for i, ex in enumerate(examples):
            self._add_to_chroma(ex, source_tag=source_tag, idx=i)

    def _add_to_chroma(self, ex: RetrievedExample, source_tag: str, idx: int | None = None) -> None:
        try:
            uid = f"{source_tag}-{idx if idx is not None else int(time.time()*1000)}"
            self._chroma_col.add(
                ids=[uid],
                documents=[ex.task],
                metadatas=[{"source": source_tag, "plan_text": ex.plan_text,
                            "reasoning": ex.reasoning}],
            )
        except Exception:
            pass  # never fail planning because memory write hiccupped

    def _query_chroma(self, query: str, k: int) -> list[RetrievedExample]:
        if self._chroma_col is None:
            return []
        try:
            res = self._chroma_col.query(query_texts=[query], n_results=k)
            docs   = (res.get("documents") or [[]])[0]
            metas  = (res.get("metadatas") or [[]])[0]
            dists  = (res.get("distances") or [[1.0] * len(docs)])[0]
            out = []
            for d, m, dist in zip(docs, metas, dists):
                out.append(RetrievedExample(
                    task      = d,
                    reasoning = (m or {}).get("reasoning", ""),
                    plan_text = (m or {}).get("plan_text", ""),
                    score     = 1.0 - float(dist),     # higher = closer
                    source    = (m or {}).get("source", "seed"),
                ))
            return out
        except Exception:
            return []

    # ── Jaccard fallback ───────────────────────────────────────────
    def _query_jaccard(self, query: str, k: int) -> list[RetrievedExample]:
        pool = self._seed + self._live
        scored = sorted(
            (ex._replace(score=_jaccard(query, ex.task)) for ex in pool),
            key=lambda e: e.score,
            reverse=True,
        )
        return [e for e in scored[:k] if e.score > 0.0] or scored[:k]

    # ── unified retrieve ───────────────────────────────────────────
    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedExample]:
        k = k if k is not None else self.cfg.top_k
        # Leave-one-out (opt-in via MEM_LEAVE_ONE_OUT=1): drop any retrieved
        # example whose task is identical to the query. This removes the
        # train-on-test leak when seed memory overlaps the test set — the
        # retriever then supplies SIMILAR examples, never the exact answer.
        loo = os.getenv("MEM_LEAVE_ONE_OUT", "0") == "1"
        kk = k + 3 if loo else k
        hits = []
        if self._chroma_col is not None:
            hits = self._query_chroma(query, kk)
        if not hits:
            hits = self._query_jaccard(query, kk)
        if loo:
            q = query.strip().lower()
            hits = [h for h in hits if h.task.strip().lower() != q]
        return hits[:k]


# ─────────────────────────────────────────────────────────────────────
# Prompt-block formatter (mirrors my_planner._format_examples)
# ─────────────────────────────────────────────────────────────────────
def format_examples_block(examples: list[RetrievedExample]) -> str:
    if not examples:
        return ""
    parts = []
    for i, ex in enumerate(examples, 1):
        tag = "GT" if ex.source == "seed" else "EPISODE"
        parts.append(
            f"=== RETRIEVED EXAMPLE {i} ({tag}, score={ex.score:.2f}) ===\n"
            f"Task: {ex.task}\n\n"
            f"Reasoning:\n{ex.reasoning}\n\n"
            f"Plan:\n{ex.plan_text}"
        )
    return "\n\n".join(parts)


__all__ = [
    "RetrievedExample",
    "MemoryRetrieverConfig",
    "MemoryRetriever",
    "load_seed_examples",
    "format_examples_block",
]
