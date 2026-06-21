#!/usr/bin/env python3
"""gen_system_figure.py
=======================
Full-system architecture block diagram for the SAGE robotics paper.

Renders the end-to-end care-robot stack as layered tiers with explicit
process / network boundaries, so a reader can see at a glance *what runs where*
and *how a natural-language command flows down to the hardware and back*:

  Interaction (dashboard / CLI / Python API)
    -- HTTP REST + WebSocket /ws/agent -->
  robot_agent runtime (UnifiedAgent + closed-loop driver + services)
    -- Planner Protocol --> pyplanner / SAGE (decompose, hybrid memory,
                            symbolic gate, suffix edit)   [+ LLM backend, HTTP]
    -- SkillRegistry.execute --> kcare_robot skills (nav / perception / manip)
    -- pyconnect (ROS2 CustomNode + agents, codecs) + VisionServe client -->
  External processes: VisionServe GPU server (TCP :11435) and the ROS2
  hardware (KAAIR arm, lift, gripper, pan-tilt head, mobile base, RGB-D cams).

The closed-loop control cycle (perceive -> plan -> map -> act -> verify ->
replan) is this system's integration contribution and is drawn as an explicit
loop; SAGE (the planner) is the paper's algorithmic contribution and is tinted
red. Nothing else is touched.

Writes: <repo_root>/ICRA/figures/system_architecture.pdf
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "font.size": 6.6,
    "font.family": "DejaVu Sans",   # has check / cross / arrow glyphs
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ---- palette ---------------------------------------------------------------
IO,   IO_E   = "#dfeede", "#3a7a3a"   # green  : human-facing / I/O
AG,   AG_E   = "#e8f0fb", "#2c5d92"   # blue   : robot_agent runtime
CL,   CL_E   = "#cfe3f7", "#1f6fb2"   # darker blue : closed-loop nodes
SV,   SV_E   = "#ede6f6", "#6a3fa0"   # purple : services
SK,   SK_E   = "#ededed", "#707070"   # gray   : skills
CN,   CN_E   = "#fdf1d6", "#b8860b"   # amber  : connectivity (pyconnect)
GPU,  GPU_E  = "#f8d3a3", "#d2730a"   # orange : VisionServe GPU server
HW,   HW_E   = "#dfe5ea", "#465562"   # slate  : ROS2 hardware
SAGE_RED     = "#c0392b"
GREEN        = "#2e8b57"
PANEL_AG     = "#f4f8fe"
PANEL_SAGE   = "#fdecea"


def box(ax, cx, cy, w, h, text, fc, ec, bold=False, fs=6.6, lw=1.0, ts=None):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.6",
                 linewidth=lw, facecolor=fc, edgecolor=ec, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            color=ts or "black",
            fontweight="bold" if bold else "normal", zorder=4)
    return {"l": (cx - w / 2, cy), "r": (cx + w / 2, cy),
            "t": (cx, cy + h / 2), "b": (cx, cy - h / 2), "c": (cx, cy)}


def panel(ax, x0, y0, x1, y1, fc, ec, title, tcolor, dashed=False, lw=1.6):
    ls = (0, (5, 3)) if dashed else "solid"
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                 boxstyle="round,pad=0.2,rounding_size=1.2",
                 facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls,
                 zorder=1))
    ax.text(x0 + 1.2, y1 - 1.0, title, ha="left", va="top", fontsize=7.0,
            fontweight="bold", color=tcolor, zorder=5)


def arr(ax, p0, p1, color="#333", cs=None, lw=1.3, style="-|>", dashed=False):
    kw = dict(arrowstyle=style, mutation_scale=9, linewidth=lw, color=color,
              shrinkA=2, shrinkB=2, zorder=6)
    if cs:
        kw["connectionstyle"] = cs
    if dashed:
        kw["linestyle"] = (0, (4, 2))
    ax.add_patch(FancyArrowPatch(p0, p1, **kw))


def boundary(ax, y, label, x0=1, x1=99):
    ax.plot([x0, x1], [y, y], color="#999", lw=0.8, ls=(0, (2, 2)), zorder=2)
    ax.text(x1 - 0.5, y + 0.5, label, ha="right", va="bottom",
            fontsize=5.9, style="italic", color="#555", zorder=5)


def main():
    root = os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".."))
    out = os.path.join(root, "ICRA", "figures", "system_architecture.pdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.3, 9.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(-9, 132)
    ax.axis("off")

    # ===================================================================
    # TIER 1 — Interaction layer
    # ===================================================================
    ax.text(50, 130.5, "Interaction layer  (optional, multi-user)",
            ha="center", va="top", fontsize=7.4, fontweight="bold",
            color=IO_E)
    b_dash = box(ax, 24, 124, 42, 8.5,
                 "Web dashboard  ·  robotapp (Next.js)\n"
                 "multi-robot · multi-location · voice I/O (vi/ko/en)\n"
                 "live plan / world-state / camera panels",
                 IO, IO_E)
    b_cli = box(ax, 60, 124, 22, 8.5,
                "CLI mode\n<pkg> name::inputs", IO, IO_E)
    b_api = box(ax, 86, 124, 22, 8.5,
                "Python API\nkcare_robot.skills.*", IO, IO_E)

    boundary(ax, 116.5,
             "HTTP REST  ·  WebSocket /ws/agent  (events + world snapshots)   |   in-process bootstrap (CLI / API)")

    # ===================================================================
    # TIER 2 — robot_agent runtime
    # ===================================================================
    panel(ax, 2, 60, 98, 114, PANEL_AG, AG_E,
          "robot_agent — FastAPI agent runtime  (robot-agnostic, single process)",
          AG_E)

    b_ua = box(ax, 50, 109, 70, 6.5,
               "UnifiedAgent  —  translate (vi/ko→en) · route  "
               "(planner = sage | direct,  plan_only)",
               AG, AG_E, bold=True)

    # ---- closed-loop driver panel ----
    panel(ax, 5, 70.5, 60, 104.5, "#eaf3fc", CL_E,
          "Closed-loop driver  (ClosedLoop)", CL_E)
    cyc = dict(fc=CL, ec=CL_E, fs=6.2)
    n_perc = box(ax, 13.5, 97, 17, 6.0, "① Perceive\nobserve", **cyc)
    n_plan = box(ax, 34, 98, 18, 6.0, "② Plan\ngenerate_plan", **cyc)
    n_map = box(ax, 50, 92.5, 16, 6.0, "③ Map\nActionMapper", **cyc)
    n_act = box(ax, 50, 79, 16, 6.0, "④ Act\nSkillRegistry", **cyc)
    n_ver = box(ax, 26, 75.5, 21, 6.0, "⑤ Verify\nisdone·symb·VLM", **cyc)
    n_rep = box(ax, 38, 86.5, 16, 6.4, "⑥ Replan\nsuffix (≤3)", **cyc)

    # forward chain (clockwise)
    arr(ax, n_perc["r"], n_plan["l"], color=CL_E)
    arr(ax, n_plan["r"], n_map["t"], color=CL_E)
    arr(ax, n_map["b"], n_act["t"], color=CL_E)
    arr(ax, n_act["l"], n_ver["r"], color=CL_E)
    # verify pass -> next step (loop up the free left edge to perceive)
    arr(ax, n_ver["l"], n_perc["b"], color=GREEN, cs="arc3,rad=0.32")
    ax.text(6.8, 86.0, "✓ pass\nnext step", ha="left", va="center",
            fontsize=5.6, color=GREEN, fontweight="bold")
    # verify fail -> replan (centre) -> splice suffix back into mapping
    arr(ax, n_ver["t"], n_rep["b"], color=SAGE_RED, cs="arc3,rad=0.2")
    ax.text(30.0, 81.2, "✗ fail", ha="center", va="center",
            fontsize=5.6, color=SAGE_RED, fontweight="bold")
    arr(ax, n_rep["t"], n_map["l"], color="#b8860b", cs="arc3,rad=-0.25")
    ax.text(45.0, 90.2, "splice\nsuffix", ha="center", va="center",
            fontsize=5.5, color="#b8860b", fontweight="bold")

    # ---- right-hand services column ----
    panel(ax, 63, 70.5, 96, 104.5, "#f3effa", SV_E,
          "Runtime services", SV_E)
    b_ws = box(ax, 79.5, 99.5, 30, 7.0,
               "WorldState  (persistent belief)\n"
               "arrived·found·holding·opened·on\n+ found_pose (stale flag)",
               SV, SV_E)
    b_vf = box(ax, 79.5, 90.5, 30, 6.6,
               "StepVerifier  (layered)\n"
               "isdone → symbolic → VLM", SV, SV_E)
    b_an = box(ax, 79.5, 82.5, 30, 5.6,
               "Announcer  ·  TTS (vi/ko/en)", SV, SV_E)
    b_rl = box(ax, 79.5, 75.5, 30, 5.6,
               "RunLogger  ·  JSONL task_runs/", SV, SV_E)
    # closed-loop reads/writes the runtime services (one tidy link)
    arr(ax, (60, 88), (63, 88), color="#888", lw=1.0, dashed=True,
        style="<|-|>")
    ax.text(61.5, 90.0, "reads /\nwrites", ha="center", va="center",
            fontsize=5.3, style="italic", color="#666")

    # ---- registries band (still inside robot_agent) ----
    b_sr = box(ax, 23, 64.5, 32, 6.2,
               "SkillRegistry\ninternal (module:func) | external (HTTP)",
               AG, AG_E)
    b_dm = box(ax, 77, 64.5, 32, 6.2,
               "DeviceManager\nconnection registry · ROS2 node owner",
               AG, AG_E)
    arr(ax, n_act["b"], b_sr["t"], color=AG_E, cs="arc3,rad=0.0")

    boundary(ax, 58.5,
             "in-process library calls  ·  pyplanner / kcare_robot")

    # ===================================================================
    # TIER 3 — Planning library (SAGE) + LLM backend
    # ===================================================================
    panel(ax, 2, 40, 64, 56.5, PANEL_SAGE, SAGE_RED,
          "★ pyplanner  ·  SAGE / SAGE   (Planner Protocol backend — this paper)",
          SAGE_RED, dashed=True, lw=2.0)
    sg = dict(fc="#f7dcd7", ec=SAGE_RED, fs=6.1)
    s1 = box(ax, 13.5, 47.5, 19, 7.5,
             "Hierarchical\ndecompose\n(sub-goals)", **sg)
    s2 = box(ax, 30.5, 47.5, 15, 7.5,
             "Hybrid memory\nseed GT + live\nChroma/Jaccard", **sg)
    s3 = box(ax, 45, 47.5, 13, 7.5,
             "Symbolic\ngate\n(0-token)", **sg, bold=True)
    s4 = box(ax, 57, 47.5, 11, 7.5,
             "Suffix\nedit\n(repair)", **sg, bold=True)
    arr(ax, s1["r"], s2["l"], color=SAGE_RED)
    arr(ax, s2["r"], s3["l"], color=SAGE_RED)
    arr(ax, s3["r"], s4["l"], color=SAGE_RED)
    arr(ax, s4["t"], s3["t"], color=SAGE_RED, cs="arc3,rad=-0.5", lw=1.0)

    b_llm = box(ax, 82, 47.5, 28, 9.0,
                "LLM backend (open-weight)\nOllama · OpenAI · Gemini\n"
                "single model, seeded",
                "#fff3cd", "#9c7a00")
    arr(ax, b_llm["l"], s2["r"], color="#9c7a00", lw=1.0, dashed=True,
        cs="arc3,rad=-0.2")
    ax.text(70.5, 52.5, "HTTP", ha="center", va="center", fontsize=5.6,
            style="italic", color="#9c7a00")
    # Planner Protocol up-link routed through the clear gutter between the two
    # registry boxes (x~49); the Plan step calls pyplanner and gets steps back.
    arr(ax, (50, 70.3), (50, 56.7), color=SAGE_RED, lw=1.3)
    ax.text(50, 63.4, "Planner Protocol\ngenerate_plan / replan",
            ha="center", va="center", fontsize=5.5, color=SAGE_RED,
            fontweight="bold", zorder=8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor=SAGE_RED, linewidth=0.6))

    # ===================================================================
    # TIER 4 — kcare_robot skills
    # ===================================================================
    panel(ax, 2, 22, 98, 38.5, "#f6f6f6", SK_E,
          "kcare_robot — robot package  (23 skills + sage_namemap + per-site configs)",
          "#555")
    b_nav = box(ax, 16, 30, 26, 9.0,
                "Navigation\nmove · moveb (Nav2)\nforward · turn · rotate\nmobile_pose",
                SK, SK_E)
    b_per = box(ax, 44, 30, 26, 9.0,
                "Perception\nfind · detect · find_arm\nget3d · grasp_succeed",
                SK, SK_E)
    b_man = box(ax, 72, 30, 26, 9.0,
                "Manipulation\npick · place · placeat\nopen/close_drawer · wipe",
                SK, SK_E)
    b_low = box(ax, 88, 30, 16, 9.0,
                "Low-level\narm · lift\nhead · grip", SK, SK_E)
    arr(ax, b_sr["b"], (27, 38.5), color=AG_E, lw=1.0)
    ax.text(30.5, 39.6, "execute(name, params, node)", ha="left", va="bottom",
            fontsize=5.6, style="italic", color="#555")

    boundary(ax, 20.5,
             "pyconnect transport   ·   node.agents[...]   ·   _vs_client()")

    # ===================================================================
    # TIER 5 — connectivity (pyconnect)
    # ===================================================================
    panel(ax, 2, 9.5, 98, 19.5, "#fff8e8", CN_E,
          "pyconnect — connectivity", CN_E)
    b_node = box(ax, 30, 14.0, 52, 6.2,
                 "ROS2 CustomNode  ·  agents (Topic / Service / Action)  ·  "
                 "msgpack codecs",
                 CN, CN_E)
    b_vsc = box(ax, 78, 14.0, 36, 6.2,
                "VisionServe client  ·  predict(model, rgb, prompt)",
                CN, CN_E)

    boundary(ax, 8.0,
             "ROS2 DDS  (topics/services/actions)        ·        TCP/HTTP :11435  (images via msgpack)")

    # ===================================================================
    # TIER 6 — external processes (hardware + GPU server)
    # ===================================================================
    panel(ax, 2, -0.5, 60, 7.0, HW, HW_E,
          "Robot hardware  ·  ROS2 process(es)", HW_E)
    box(ax, 16, 2.7, 26, 4.6,
        "KAAIR 6-DOF arm · lift\n2-finger + suction gripper",
        "#eef1f4", HW_E, fs=6.0)
    box(ax, 45, 2.7, 28, 4.6,
        "Pan-tilt head · mobile base (Nav2)\nD405 wrist + Femto Bolt head RGB-D",
        "#eef1f4", HW_E, fs=6.0)

    panel(ax, 63, -0.5, 98, 7.0, GPU, GPU_E,
          "VisionServe — GPU inference server  (separate process)", GPU_E)
    box(ax, 80.5, 2.7, 32, 4.6,
        "GroundingDINO · GroundedSAM\ngrasp-gd  (open-vocab + grasp)",
        "#fde9d0", GPU_E, fs=6.0)

    arr(ax, b_node["b"], (30, 7.0), color=CN_E, lw=1.1)
    arr(ax, b_vsc["b"], (80.5, 7.0), color=GPU_E, lw=1.1)

    # ===================================================================
    # legend
    # ===================================================================
    lx, ly = 6.0, -6.0
    items = [
        (IO, IO_E, "human / I/O"),
        (AG, AG_E, "agent runtime"),
        (SAGE_RED, SAGE_RED, "SAGE (ours)"),
        (SK, SK_E, "robot skills"),
        (CN, CN_E, "connectivity"),
        (GPU, GPU_E, "GPU server"),
        (HW, HW_E, "hardware"),
    ]
    for i, (fc, ec, lab) in enumerate(items):
        x = lx + i * 13.5
        ax.add_patch(FancyBboxPatch((x, ly), 2.2, 1.8,
                     boxstyle="round,pad=0.02,rounding_size=0.3",
                     facecolor=fc, edgecolor=ec, linewidth=0.8, zorder=7))
        ax.text(x + 2.9, ly + 0.9, lab, ha="left", va="center",
                fontsize=5.8, zorder=7)

    fig.savefig(out, bbox_inches="tight", pad_inches=0.05, dpi=200)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
