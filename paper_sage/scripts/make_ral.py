#!/usr/bin/env python3
"""Reproducibly derive the RA-L 8-page version from the canonical ICRA sources.

Reads ICRA/tex/experiment.tex + ICRA/tex/intro_related.tex (the FULL canonical
sources) and writes the trimmed ICRA/ral/tex/experiment.tex +
ICRA/ral/tex/intro_related.tex, applying the agreed cuts (B1 scaling, B5 pareto,
B10 pstrict, B2 predictive, B3 procthor, B4 sota-compress, B6 positioning,
autoverifier->supp, Models-trim). method.tex/conclusion.tex are copied verbatim.
Idempotent; safe to re-run after the canonical sections are edited, as long as
edits do not rewrite the exact cut anchors (subsection titles / table labels).
"""
import re, os, shutil

ICRA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ICRA")

def rm_float(s, env, label):
    pat = re.compile(r"\\begin\{"+env+r"\}(?:(?!\\end\{"+env+r"\}).)*?"+re.escape(label)+r"(?:(?!\\end\{"+env+r"\}).)*?\\end\{"+env+r"\}\s*", re.S)
    s2 = pat.sub("", s, count=1)
    if s2 == s: print(f"  WARN: {env}/{label} not found")
    return s2

# ---------- experiment.tex ----------
e = open(os.path.join(ICRA, "tex/experiment.tex")).read()

# B1 scaling subsection
i = e.find("\\subsection{Scaling across model size}")
j = e.find("\\subsection{Stressing the saturated metric")
if i!=-1 and j!=-1: e = e[:i] + e[j:]
else: print("  WARN: B1 scaling anchors not found")

# B5 pareto figure + refs
e = re.sub(r"\\begin\{figure\}(?:(?!\\end\{figure\}).)*?cost_quality_pareto(?:(?!\\end\{figure\}).)*?\\end\{figure\}\s*", "", e, count=1, flags=re.S)
e = e.replace("Table~\\ref{tab:cost} and\nFig.~\\ref{fig:pareto} place", "Table~\\ref{tab:cost} places")
e = e.replace(" and\nFig.~\\ref{fig:pareto}", "").replace("Fig.~\\ref{fig:pareto} ", "").replace(" Fig.~\\ref{fig:pareto}", "")

# B10 pstrict table* + reword its sentence
e = rm_float(e, "table\\*", "tab:pstrict")
e = e.replace(
 "We report\n\\texttt{precondition\\_strict} (Table~\\ref{tab:pstrict}) only as a diagnostic,\nsince it is computed by SAGE's own verifier and is therefore circular as a\nquality score.",
 "We report \\texttt{precondition\\_strict} only as a circular diagnostic (it is computed by SAGE's own verifier; per-model values are in the supplement).")

# B2 O-pred paragraph + its predictive table. Stop at the NEXT paragraph/subsection
# so we never swallow a following must-stay block (e.g. Failure recovery).
m = re.search(r"\\paragraph\*\{\(O-pred\).*?(?=\n\\paragraph|\n\\subsection|\Z)", e, re.S)
if m: e = e[:m.start()] + e[m.end():]
else: print("  WARN: B2 O-pred not found")
e = rm_float(e, "table", "tab:predictive")

# B3 ProcTHOR paragraph -> one sentence, remove table
m = re.search(r"\\paragraph\*\{Unseen procedurally-generated houses\.\}.*?(?=\n\\paragraph|\n\\subsection|\Z)", e, re.S)
if m:
    e = e[:m.start()] + ("\\paragraph*{Unseen procedurally-generated houses.}\n"
        "On 70 unseen ProcTHOR houses, with identical code and seed memory, SAGE retains its strict-precondition lead ($0.985$): plan \\emph{validity} transfers to never-seen layouts (full table in the supplement).\n") + e[m.end():]
e = rm_float(e, "table", "tab:procthor")

# B4 SOTA subsection -> compact paragraph (+ remove table)
e = rm_float(e, "table", "tab:sota")
m = re.search(r"\\subsection\{Comparison to published baselines\}\\label\{sec:sota\}.*?(?=\n\\subsection)", e, re.S)
if m:
    e = e[:m.start()] + ("\\paragraph*{Published baselines.}\n"
        "Against two established planners --- \\textbf{LLM+P}~\\cite{liu2023llmp} "
        "(LLM-to-PDDL solved by a classical planner) and a \\textbf{SayCan}-style "
        "affordance-greedy decoder~\\cite{ahn2022saycan} --- both produce \\emph{valid but "
        "incomplete} plans (completeness $0.62$--$0.77$ on 7B/14B) whereas SAGE reaches "
        "$0.93$--$0.96$; SayCan's strict-precondition is $1.0$ \\emph{by construction}, as "
        "it uses our verifier as its affordance filter. Full per-model numbers are in the supplement.\n\n") + e[m.end():]
else: print("  WARN: B4 sota subsection not found")

# autoverifier table -> supp
e = rm_float(e, "table", "tab:autoverify")
e = e.replace("(Table~\\ref{tab:autoverify})", "(supplement)").replace("Table~\\ref{tab:autoverify}", "the supplement")

# RA-L only: move failure-taxonomy table to supp (keep the 17x claim as prose),
# reword the two \ref{tab:failtax} so no dangling reference remains.
e = rm_float(e, "table", "tab:failtax")
# Robustly drop every \ref{tab:failtax} (the table now lives in the supplement),
# tolerant of whatever surrounding wording the prose currently uses.
e = re.sub(r"Table~\\ref\{tab:failtax\}\s+makes this\s+explicit",
           "The failure taxonomy (supplement) makes this explicit", e)
# NB: use lambda replacements so backslashes (\ref) are NOT escape-processed by
# re.sub (a plain "(Table~\\ref{tab:sim})" replacement turns \r into a carriage
# return -> the "Table eftab:sim" corruption).
e = re.sub(r"\(\s*Tables?~\\ref\{tab:sim\},\s*~?\\ref\{tab:failtax\}\s*\)",
           lambda m: "(Table~\\ref{tab:sim})", e)
e = re.sub(r",?\s*~?\\ref\{tab:failtax\}", lambda m: "", e)  # catch-all leftover

# RA-L only: move the cost table to the supplement too (the ~7x claim stays in
# prose; the pareto figure is already removed by B5 above, so drop the now-empty
# "...places SAGE on the cost--quality frontier" sentence).
e = rm_float(e, "table", "tab:cost")
e = re.sub(r"Table~\\ref\{tab:cost\} places SAGE on the cost--quality frontier\.\s*", "", e)
e = re.sub(r",?\s*~?\\ref\{tab:cost\}", lambda m: "", e)  # catch-all leftover

# Models paragraph trim
m = re.search(r"\\paragraph\*\{Models\.\}.*?(?=\n\\paragraph)", e, re.S)
if m:
    e = e[:m.start()] + ("\\paragraph*{Models.}\n"
        "We benchmark five open-weight Ollama models spanning a $\\sim\\!10\\times$ parameter "
        "range --- \\texttt{llama3.2}~(3B), \\texttt{qwen2.5:7b}, \\texttt{mistral-nemo}~(12B), "
        "\\texttt{qwen2.5:14B} and \\texttt{qwen2.5:32b} --- via the same backend; only the "
        "model identifier differs, guarding against single-model artifacts.\n\n") + e[m.end():]

open(os.path.join(ICRA, "ral/tex/experiment.tex"), "w").write(e)

# ---------- intro_related.tex (B6 positioning table) ----------
s = open(os.path.join(ICRA, "tex/intro_related.tex")).read()
s = re.sub(r"\\begin\{table\}(?:(?!\\end\{table\}).)*?tab:positioning(?:(?!\\end\{table\}).)*?\\end\{table\}\s*", "", s, count=1, flags=re.S)
s = re.sub(r"Table~\\ref\{tab:positioning\} situates SAGE along the axes that\s*distinguish it\. ", "", s)
s = s.replace("\\ref{tab:positioning}", "the axes above")
open(os.path.join(ICRA, "ral/tex/intro_related.tex"), "w").write(s)

# ---------- shared sections copied verbatim ----------
for f in ("method.tex", "conclusion.tex"):
    shutil.copy(os.path.join(ICRA, "tex", f), os.path.join(ICRA, "ral/tex", f))

# ---------- ral/main.tex from canonical main.tex ----------
# \input{\sectiondir/...} (sectiondir=tex) resolves to ral/tex/* when built from ral/.
# RA-L-specific: 10pt option, the full 7-author block (3 added at positions 4-6),
# and vspace tightening to avoid a cram-full last page.
mm = open(os.path.join(ICRA, "main.tex")).read()
mm = mm.replace("\\documentclass[conference]{IEEEtran}",
                "\\documentclass[10pt,conference]{IEEEtran}")

RAL_VSPACE = (
    "\\usepackage[hidelinks]{hyperref}\n"
    "% --- RA-L space tightening (float/caption/display) ---\n"
    "\\setlength{\\textfloatsep}{5pt plus 2pt minus 2pt}\n"
    "\\setlength{\\floatsep}{5pt plus 2pt minus 2pt}\n"
    "\\setlength{\\intextsep}{5pt plus 2pt minus 2pt}\n"
    "\\setlength{\\dbltextfloatsep}{6pt plus 2pt minus 2pt}\n"
    "\\setlength{\\abovecaptionskip}{3pt}\n"
    "\\setlength{\\belowcaptionskip}{0pt}\n"
    "\\setlength{\\abovedisplayskip}{3pt plus 1pt minus 1pt}\n"
    "\\setlength{\\belowdisplayskip}{3pt plus 1pt minus 1pt}\n"
    "\\setlength{\\abovedisplayshortskip}{1pt plus 1pt}\n"
    "\\setlength{\\belowdisplayshortskip}{2pt plus 1pt minus 1pt}\n"
)
mm = mm.replace("\\usepackage[hidelinks]{hyperref}\n", RAL_VSPACE, 1)

# 7-author block (Phung/Jun/Hwang inserted at positions 4-6; Shin -> 7)
mm = mm.replace(
    "\\IEEEauthorblockN{Anonymous Author\\IEEEauthorrefmark{1},\n"
    "Anonymous Author\\IEEEauthorrefmark{1},\n"
    "Anonymous Author\\IEEEauthorrefmark{1},\n"
    "and Anonymous Author\\IEEEauthorrefmark{1}\\IEEEauthorrefmark{2}}",
    "\\IEEEauthorblockN{Anonymous Author\\IEEEauthorrefmark{1},\n"
    "Anonymous Author\\IEEEauthorrefmark{1},\n"
    "Anonymous Author\\IEEEauthorrefmark{1},\n"
    "Anonymous Author\\IEEEauthorrefmark{1},\\\\\n"
    "Anonymous Author\\IEEEauthorrefmark{1},\n"
    "Anonymous Author\\IEEEauthorrefmark{1},\n"
    "and Anonymous Author\\IEEEauthorrefmark{1}\\IEEEauthorrefmark{2}}")
mm = mm.replace(
    "\\IEEEauthorblockA{\\{anon\\}@anon.invalid, di\\_shin@anon.invalid}}",
    "\\IEEEauthorblockA{\\{anon\\}@anon.invalid, di\\_shin@anon.invalid}}")

open(os.path.join(ICRA, "ral/main.tex"), "w").write(mm)

# ---------- sync assets the RA-L build needs (tables + figures) ----------
os.makedirs(os.path.join(ICRA, "ral/tables"), exist_ok=True)
os.makedirs(os.path.join(ICRA, "ral/figures"), exist_ok=True)
for fn in os.listdir(os.path.join(ICRA, "tables")):
    if fn.endswith(".tex"):
        shutil.copy(os.path.join(ICRA, "tables", fn), os.path.join(ICRA, "ral/tables", fn))
# figures the RA-L body actually references: impact (intro), pipeline (method),
# difficulty_bars (compound). Copy all current figure PDFs to stay safe.
for fn in os.listdir(os.path.join(ICRA, "figures")):
    if fn.endswith((".pdf", ".png")):
        shutil.copy(os.path.join(ICRA, "figures", fn), os.path.join(ICRA, "ral/figures", fn))

print("make_ral: wrote ral/main.tex + ral/tex/* + synced ral/tables + ral/figures")
