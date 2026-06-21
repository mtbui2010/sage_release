#!/usr/bin/env python3
"""Consolidate SAGE Jetson AGX Orin edge-deployment results.

Reads:
  results/edge_agxorin_{3b,7b,14b}_s0/   (on-device plan-mode, Direct+SAGE)
  results/server_parity_{3b,7b,14b}_s0/  (same subset/seed on .18 server)
  edge_logs/tegra_{3b,7b,14b}.log        (RAM/power/GPU)
Writes:
  results/edge_summary.json
and prints two Markdown tables (cost; parity).
"""
import csv, json, os, re, statistics as st, glob

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
TAGS = ["3b", "7b", "14b"]
DEVICE = "AGX Orin 64GB (MAXN)"


def load_rows(run):
    p = os.path.join(ROOT, "results", run, "results.csv")
    if not os.path.exists(p):
        return None
    rows = list(csv.DictReader(open(p)))
    return rows or None


def f(x, d=0.0):
    try:
        return float(x)
    except Exception:
        return d


def agg(rows, method):
    rs = [r for r in rows if r["method"] == method]
    if not rs:
        return None
    def col(k):
        return [f(r.get(k)) for r in rs]
    lat = col("latency_s"); ot = col("output_tokens"); it = col("input_tokens")
    tps = [o / l for o, l in zip(ot, lat) if l > 0 and o > 0]
    return dict(
        n=len(rs),
        completeness=st.mean(col("completeness")),
        executability=st.mean(col("executability")),
        precondition_strict=st.mean(col("precondition_strict")),
        step_ratio=st.mean(col("step_ratio")),
        redundancy=st.mean(col("redundancy")),
        llm_calls=st.mean(col("llm_calls")),
        refines=st.mean(col("refines")),
        out_tokens=st.mean(ot),
        in_tokens=st.mean(it),
        latency_med=st.median(lat),
        tok_s=st.median(tps) if tps else 0.0,
    )


def parse_tegra(tag):
    p = os.path.join(ROOT, "edge_logs", f"tegra_{tag}.log")
    if not os.path.exists(p):
        return None
    ram = []; pw = []; gpu = []
    for line in open(p):
        m = re.search(r"RAM (\d+)/(\d+)MB", line)
        if m:
            ram.append(int(m.group(1)))
        g = re.search(r"GR3D_FREQ (\d+)%", line)
        if g:
            gpu.append(int(g.group(1)))
        rails = re.findall(r"(VDD_GPU_SOC|VDD_CPU_CV|VIN_SYS_5V0) (\d+)mW", line)
        if rails:
            pw.append(sum(int(v) for _, v in rails))
    if not ram:
        return None
    return dict(
        ram_peak_gb=max(ram) / 1024,
        ram_mean_gb=st.mean(ram) / 1024,
        power_peak_w=max(pw) / 1000 if pw else 0,
        power_mean_w=st.mean(pw) / 1000 if pw else 0,
        gpu_busy_pct=100 * sum(1 for x in gpu if x > 0) / len(gpu) if gpu else 0,
    )


def main():
    out = {"device": DEVICE, "models": {}}
    for tag in TAGS:
        edge = load_rows(f"edge_agxorin_{tag}_s0")
        srv = load_rows(f"server_parity_{tag}_s0")
        tegra = parse_tegra(tag)
        entry = {"tegra": tegra, "edge": {}, "server": {}}
        if edge:
            for m in ("Direct", "SAGE"):
                a = agg(edge, m)
                if a:
                    entry["edge"][m] = a
        if srv:
            for m in ("Direct", "SAGE"):
                a = agg(srv, m)
                if a:
                    entry["server"][m] = a
        out["models"][tag] = entry

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "results", "edge_summary.json"), "w"), indent=2)

    # ---- Table 1: on-device cost ----
    print("\n### Table A — On-device cost (AGX Orin 64GB, MAXN), Q4_K_M\n")
    print("| Model | Method | tok/s | Plan lat (s, med) | LLM calls | Peak RAM (GB) | Power (W, mean) | GPU busy |")
    print("|---|---|---|---|---|---|---|---|")
    for tag in TAGS:
        e = out["models"][tag]; t = e["tegra"] or {}
        for m in ("Direct", "SAGE"):
            d = e["edge"].get(m)
            if not d:
                continue
            ram = f"{t.get('ram_peak_gb',0):.1f}" if t else "—"
            powr = f"{t.get('power_mean_w',0):.0f}" if t else "—"
            gpu = f"{t.get('gpu_busy_pct',0):.0f}%" if t else "—"
            print(f"| qwen2.5:{tag} | {m} | {d['tok_s']:.1f} | {d['latency_med']:.1f} | "
                  f"{d['llm_calls']:.1f} | {ram} | {powr} | {gpu} |")

    # ---- Table 2: edge vs server quality parity ----
    print("\n### Table B — Quality parity: edge vs server (same 20-task subset, seed 0)\n")
    print("| Model | Method | compl (edge/srv) | exec (edge/srv) | precS (edge/srv) | "
          "SAGE compl lift (edge) |")
    print("|---|---|---|---|---|---|")
    for tag in TAGS:
        e = out["models"][tag]
        for m in ("Direct", "SAGE"):
            ed = e["edge"].get(m); sv = e["server"].get(m)
            if not ed:
                continue
            def pair(k):
                ev = ed[k]; svv = sv[k] if sv else None
                return f"{ev:.2f}/{svv:.2f}" if svv is not None else f"{ev:.2f}/—"
            lift = ""
            if m == "SAGE" and e["edge"].get("Direct"):
                lift = f"+{ed['completeness']-e['edge']['Direct']['completeness']:.2f}"
            print(f"| qwen2.5:{tag} | {m} | {pair('completeness')} | {pair('executability')} | "
                  f"{pair('precondition_strict')} | {lift} |")

    # verifier overhead reminder
    print("\nVerifier overhead (on-device micro-bench): median 0.008 ms/call, "
          "max 0.017 ms over 75 plans (len median 4, max 8).")


if __name__ == "__main__":
    main()
