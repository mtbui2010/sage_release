"""plot_results.py
====================
Generate the figures referenced by the SAGE paper from a results CSV.

Figures emitted (PNG + PDF), all under figures/{run_id}/:

  1. cost_quality_pareto.{png,pdf}
        Scatter of (mean total_tokens, mean precondition score) for each
        (model, method). SAGE should occupy the top-left.

  2. per_metric_bars.{png,pdf}
        Bar chart per method for: precondition, executability, step_ratio,
        hallucination. Faceted by model.

  3. ablation.{png,pdf}
        Same per-metric bars but restricted to {SAGE, SAGE-NoVerifier,
        SAGE-NoRepair, SAGE-NoMemory}.

  4. token_breakdown.{png,pdf}
        Stacked bars: input_tokens vs output_tokens per method.

Dependencies: matplotlib, pandas. Skips gracefully if not installed.

Usage
-----
  python scripts/plot_results.py results/<run_id>/results.csv
  python scripts/plot_results.py results/<run_id>/results.csv --out figures/<run_id>/

"""
from __future__ import annotations

import argparse
import os
import sys


def _require():
    try:
        import pandas as pd  # noqa
        import matplotlib    # noqa
        import matplotlib.pyplot as plt  # noqa
        # Defaults that play nicely with the .eps backend (no transparency,
        # no missing-glyph warnings for sans-serif Unicode):
        matplotlib.rcParams["figure.facecolor"] = "white"
        matplotlib.rcParams["savefig.facecolor"] = "white"
        matplotlib.rcParams["savefig.transparent"] = False
        matplotlib.rcParams["ps.fonttype"] = 42       # embed TrueType in EPS
        matplotlib.rcParams["pdf.fonttype"] = 42
        return pd, plt
    except ImportError as e:
        print(f"[ERROR] missing dependency: {e}. Install with:")
        print("  pip install pandas matplotlib")
        sys.exit(1)


def _merge_extras(pd, base_csv: str, extra_paths: list[str]):
    """Load base CSV plus one-or-more extras, replacing any (model, method)
    pair found in an EXTRA with the EXTRA's rows.

    Use case: re-ran one method (e.g. Few-Shot CoT after a parser fix) into a
    new run; want the figure to show that fresh data alongside the unchanged
    rows of the original run, without re-running everything.

    Later extras win over earlier ones — pass them in increasing order of
    freshness. Stale rows in the base for matching (model, method) keys are
    silently dropped; a one-line summary is printed per extra.
    """
    base = pd.read_csv(base_csv)
    if not extra_paths:
        return base
    base["__src__"] = "base"
    merged = base.copy()
    for ext in extra_paths:
        if not os.path.exists(ext):
            print(f"[WARN] --extra path not found, skipping: {ext}")
            continue
        ex = pd.read_csv(ext)
        if ex.empty:
            print(f"[WARN] --extra CSV is empty, skipping: {ext}")
            continue
        ex["__src__"] = f"extra:{os.path.basename(os.path.dirname(ext)) or ext}"
        pairs = set(map(tuple, ex[["model", "method"]].drop_duplicates().values))
        before = len(merged)
        if pairs:
            mask = list(zip(merged["model"], merged["method"]))
            keep = [k not in pairs for k in mask]
            merged = merged[keep].copy()
        dropped = before - len(merged)
        merged = pd.concat([merged, ex], ignore_index=True)
        pretty = ", ".join(f"({m},{me})" for m, me in sorted(pairs))
        print(f"[INFO] merged extra {ext}: dropped {dropped} stale row(s) "
              f"in base, added {len(ex)} fresh row(s) for {pretty}")
    return merged


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("csv", help="Path to base results.csv produced by run_benchmark.py")
    p.add_argument("--extra", nargs="*", default=[],
                   help="One or more additional results.csv files whose (model, method) "
                        "rows REPLACE the corresponding rows in the base CSV. Useful for "
                        "re-running a single method after a bug fix.")
    p.add_argument("--out", default="", help="Output dir (default: figures/<csv stem>/)")
    p.add_argument("--style", default="seaborn-v0_8", help="matplotlib style sheet")
    args = p.parse_args()

    pd, plt = _require()
    try:
        plt.style.use(args.style)
    except Exception:
        pass

    df_all = _merge_extras(pd, args.csv, args.extra)
    if df_all.empty:
        print("[ERROR] empty results CSV")
        return 1

    # Compute error rate per (model, method) BEFORE dropping error rows so
    # the errors_breakdown figure is grounded in the full dataset.
    df_all["has_error"] = (df_all.get("error", "").fillna("") != "").astype(int)
    err = (df_all.groupby(["model", "method"])["has_error"]
                 .agg(["sum", "count"]).reset_index()
                 .rename(columns={"sum": "errors", "count": "n"}))
    err["error_rate"] = err["errors"] / err["n"]
    n_total   = len(df_all)
    n_errored = int(df_all["has_error"].sum())

    df = df_all[df_all["has_error"] == 0].copy()
    if df.empty:
        print("[ERROR] every row has an error — nothing to plot")
        print(f"        ({n_errored}/{n_total} rows had an error)")
        return 1
    if n_errored:
        print(f"[INFO] dropped {n_errored}/{n_total} rows that had errors")

    out_dir = args.out or os.path.join(
        os.path.dirname(args.csv), "..", "..", "figures",
        os.path.basename(os.path.dirname(args.csv))
    )
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Writing figures to {out_dir}")

    # precondition_strict may be absent in older CSVs — default to 0 so the
    # aggregate doesn't crash, and skip the related panels if all-zero.
    if "precondition_strict" not in df.columns:
        df["precondition_strict"] = 0.0

    agg = df.groupby(["model", "method"]).agg({
        "total_tokens":         "mean",
        "latency_s":            "mean",
        "step_ratio":           "mean",
        "executability":        "mean",
        "precondition":         "mean",
        "precondition_strict":  "mean",
        "redundancy":           "mean",
        "completeness":         "mean",
        "hallucination":        "mean",
        "llm_calls":            "mean",
        "input_tokens":         "mean",
        "output_tokens":        "mean",
        "refines":              "mean",
        "num_steps":            "mean",
    }).reset_index()

    # Methods whose quality is 0 across the board → upstream parse failure.
    # We mark these on the x-axis instead of hiding them, so a reader of the
    # figure can tell "this baseline is degenerate, not just bad".
    quality_cols = ["precondition", "executability", "completeness", "num_steps"]
    parse_fail_pairs = set()
    for _, row in agg.iterrows():
        if all(float(row.get(c, 0.0)) == 0.0 for c in quality_cols):
            parse_fail_pairs.add((row["model"], row["method"]))
    if parse_fail_pairs:
        bad = sorted({m for _, m in parse_fail_pairs})
        print(f"[WARN] degenerate (zero-quality) methods detected: {bad}")
        print("       → flagged as 'method (parse_fail)' in figures")

    def _label(model, method):
        return f"{method} (parse_fail)" if (model, method) in parse_fail_pairs else method

    # ── 1. Cost-quality Pareto ───────────────────────────────────
    # We draw one panel per model so labels never overlap across models.
    models = sorted(agg["model"].unique())
    n_models = len(models)
    fig, axes = plt.subplots(
        1, n_models,
        figsize=(5.5 * n_models, 5),
        sharey=True,
    )
    if n_models == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        sub = agg[agg["model"] == model]
        ax.scatter(sub["total_tokens"], sub["precondition"], s=80,
                   color="#3470b3", edgecolor="black", linewidth=0.5)
        # Light vertical jitter on annotation offset so coincident points
        # (e.g. Direct and CoT both at precond=1.0) stay readable.
        used_y: list[tuple[float, float]] = []
        for _, row in sub.iterrows():
            lbl = _label(row["model"], row["method"])
            x, y = float(row["total_tokens"]), float(row["precondition"])
            offset_y = 6
            for ux, uy in used_y:
                if abs(ux - x) < 600 and abs(uy - y) < 0.05:
                    offset_y += 11
            used_y.append((x, y))
            ax.annotate(
                lbl, (x, y),
                fontsize=8, xytext=(5, offset_y),
                textcoords="offset points",
            )
        ax.set_xlabel("Mean total tokens per task (lower is better)")
        if ax is axes[0]:
            ax.set_ylabel("Mean precondition score (higher is better)")
        ax.set_title(f"model = {model}")
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.05, 1.15)
    fig.suptitle("Cost–quality Pareto across planning methods")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "pdf", "eps"):
        fig.savefig(os.path.join(out_dir, f"cost_quality_pareto.{ext}"), dpi=150)
    plt.close(fig)

    # ── 2. Per-metric bars ───────────────────────────────────────
    metrics = ["precondition", "executability", "step_ratio", "hallucination"]
    for model, sub in agg.groupby("model"):
        sub = sub.sort_values("method").copy()
        sub["label"] = [_label(model, m) for m in sub["method"]]
        fig, axes = plt.subplots(
            1, len(metrics),
            figsize=(5.5 * len(metrics), 5),
        )
        for ax, m in zip(axes, metrics):
            colors = ["#bbbbbb" if "parse_fail" in lab else "#3470b3"
                      for lab in sub["label"]]
            bars = ax.bar(sub["label"], sub[m], color=colors,
                          edgecolor="black", linewidth=0.5)
            ax.set_title(m, fontsize=11)
            ax.set_ylabel("mean")
            ax.tick_params(axis="x", rotation=60, labelsize=8)
            for label in ax.get_xticklabels():
                label.set_ha("right")
            # Annotate zero-quality bars so a reader cannot miss them.
            for rect, lab in zip(bars, sub["label"]):
                if "parse_fail" in lab:
                    ax.text(rect.get_x() + rect.get_width() / 2,
                            0.02 * (ax.get_ylim()[1] or 1),
                            "0",
                            ha="center", va="bottom",
                            fontsize=8, color="#666666")
        fig.suptitle(f"Per-metric bars — model={model}")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        suffix = model.replace(":", "_").replace("/", "_")
        for ext in ("png", "pdf", "eps"):
            fig.savefig(os.path.join(out_dir, f"per_metric_bars_{suffix}.{ext}"), dpi=150)
        plt.close(fig)

    # ── 3. Ablation ──────────────────────────────────────────────
    # An ablation chart only makes sense if multiple SAGE variants exist.
    # Warn loudly when only "SAGE" is present (i.e. user ran `make benchmark`
    # but never `make ablate`).
    abl_methods = sorted({m for m in agg["method"] if m.startswith("SAGE")})
    if len(abl_methods) < 2:
        print(
            f"[WARN] only {len(abl_methods)} SAGE variant(s) found in this run: {abl_methods}\n"
            f"       → ablation figures need {{SAGE, SAGE-NoVerifier, "
            f"SAGE-NoRepair, SAGE-NoMemory}}.\n"
            f"       → run 'make ablate' to produce a meaningful ablation chart."
        )
    else:
        abl = agg[agg["method"].str.startswith("SAGE")]
        for model, sub in abl.groupby("model"):
            sub = sub.sort_values("method")
            fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
            for ax, m in zip(axes, metrics):
                ax.bar(sub["method"], sub[m], color="C2")
                ax.set_title(m)
                ax.tick_params(axis="x", rotation=30)
            fig.suptitle(f"SAGE ablation — model={model}")
            fig.tight_layout()
            suffix = model.replace(":", "_").replace("/", "_")
            for ext in ("png", "pdf", "eps"):
                fig.savefig(os.path.join(out_dir, f"ablation_{suffix}.{ext}"), dpi=150)
            plt.close(fig)

    # ── 4. Token breakdown ───────────────────────────────────────
    for model, sub in agg.groupby("model"):
        sub = sub.sort_values("method").copy()
        sub["label"] = [_label(model, m) for m in sub["method"]]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(sub["label"], sub["input_tokens"], label="input",  color="#3470b3")
        ax.bar(sub["label"], sub["output_tokens"], bottom=sub["input_tokens"],
               label="output", color="#f0a040")
        ax.set_ylabel("Mean tokens per task")
        ax.set_title(f"Token breakdown — model={model}")
        ax.tick_params(axis="x", rotation=60, labelsize=8)
        for label in ax.get_xticklabels():
            label.set_ha("right")
        ax.legend()
        fig.tight_layout()
        suffix = model.replace(":", "_").replace("/", "_")
        for ext in ("png", "pdf", "eps"):
            fig.savefig(os.path.join(out_dir, f"token_breakdown_{suffix}.{ext}"), dpi=150)
        plt.close(fig)

    # ── 4b. Precondition: heuristic vs strict (side-by-side) ────
    # Surfaces the gap between the legacy "Find/Nav precedes interact"
    # heuristic and the symbolic-verifier-grounded strict score. Methods
    # whose strict score collapses while the heuristic stays at 1.0 are
    # silently producing invalid plans.
    if agg["precondition_strict"].sum() > 0:
        import numpy as _np
        for model, sub in agg.groupby("model"):
            sub = sub.sort_values("method").copy()
            sub["label"] = [_label(model, m) for m in sub["method"]]
            x = _np.arange(len(sub))
            width = 0.4
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(x - width / 2, sub["precondition"], width,
                   label="precondition (heuristic)", color="#9bbcdb")
            ax.bar(x + width / 2, sub["precondition_strict"], width,
                   label="precondition_strict (verifier)", color="#1f4f82")
            ax.set_xticks(x)
            ax.set_xticklabels(sub["label"], rotation=60, ha="right", fontsize=8)
            ax.set_ylabel("Mean score (0–1)")
            ax.set_ylim(0, 1.05)
            ax.set_title(f"Heuristic vs strict precondition — model={model}")
            ax.legend(loc="lower right")
            fig.tight_layout()
            suffix = model.replace(":", "_").replace("/", "_")
            for ext in ("png", "pdf", "eps"):
                fig.savefig(os.path.join(out_dir, f"precondition_strict_vs_heuristic_{suffix}.{ext}"), dpi=150)
            plt.close(fig)
        print("[INFO] wrote precondition_strict_vs_heuristic_*.png")
    else:
        print("[INFO] precondition_strict all zero — skipping verifier-vs-heuristic figure")

    # ── 5. Errors breakdown ──────────────────────────────────────
    # Counts per (model, method) of how many tasks raised an exception.
    # Hidden until now because users seldom inspect the CSV directly.
    if int(err["errors"].sum()) > 0:
        for model, sub in err.groupby("model"):
            sub = sub.sort_values("method").copy()
            sub["label"] = [_label(model, m) for m in sub["method"]]
            fig, ax = plt.subplots(figsize=(10, 5))
            colors = ["#d62728" if e > 0 else "#dddddd" for e in sub["errors"]]
            ax.bar(sub["label"], sub["errors"], color=colors,
                   edgecolor="black", linewidth=0.5)
            ax.set_ylabel("Tasks that raised an exception")
            ax.set_title(f"Errors per method — model={model}")
            ax.tick_params(axis="x", rotation=60, labelsize=8)
            for label in ax.get_xticklabels():
                label.set_ha("right")
            for x, (e, n) in enumerate(zip(sub["errors"], sub["n"])):
                if e > 0:
                    ax.text(x, e + 0.05, f"{int(e)}/{int(n)}",
                            ha="center", fontsize=8)
            fig.tight_layout()
            suffix = model.replace(":", "_").replace("/", "_")
            for ext in ("png", "pdf", "eps"):
                fig.savefig(os.path.join(out_dir, f"errors_breakdown_{suffix}.{ext}"), dpi=150)
            plt.close(fig)
        print(f"[INFO] wrote errors_breakdown_*.png ({int(err['errors'].sum())} total errors)")
    else:
        print("[INFO] no errors in any row — skipping errors_breakdown figure")

    print(f"All figures written under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
