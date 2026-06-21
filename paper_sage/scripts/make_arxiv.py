#!/usr/bin/env python3
"""Reproducibly derive the arXiv version (main + supplement-as-appendix) from the
canonical ICRA sources. Regenerates ICRA/arxiv/main.tex, ICRA/arxiv/supp_appendix.tex,
syncs ICRA/arxiv/tex/*, and copies shared assets. Run after canonical edits.
"""
import os, shutil

ICRA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ICRA")
AX = os.path.join(ICRA, "arxiv")
os.makedirs(os.path.join(AX, "tex"), exist_ok=True)
os.makedirs(os.path.join(AX, "tables"), exist_ok=True)
os.makedirs(os.path.join(AX, "supp_tables"), exist_ok=True)
os.makedirs(os.path.join(AX, "figures"), exist_ok=True)

# 1) supp_appendix.tex = canonical supplement body (after \end{abstract} .. before \end{document})
supp = open(os.path.join(ICRA, "supplementary.tex")).read().splitlines(keepends=True)
ea = next(i for i, l in enumerate(supp) if "\\end{abstract}" in l)
ed = next(i for i, l in enumerate(supp) if "\\end{document}" in l)
open(os.path.join(AX, "supp_appendix.tex"), "w").write("".join(supp[ea+1:ed]).strip() + "\n")

# 2) arxiv/main.tex = canonical main.tex + \suppinput macro + appendix before \end{document}
out = []
for l in open(os.path.join(ICRA, "main.tex")):
    if l.startswith("\\newcommand{\\sectiondir}"):
        out.append(l)
        out.append("% --- arXiv: supplement-as-appendix support ---\n")
        out.append("\\newcommand{\\suppinput}[1]{\\IfFileExists{#1}{\\input{#1}}{\\textit{[missing: \\texttt{#1}]}}}\n")
        continue
    if l.strip() == "\\end{document}":
        out.append("\n% ===== SUPPLEMENTARY MATERIAL (arXiv appendix) =====\n")
        out.append("\\clearpage\n\\appendix\n")
        out.append("\\begin{center}\\large\\textbf{Supplementary Material}\\end{center}\n")
        out.append("\\setcounter{table}{0}\\renewcommand{\\thetable}{S\\arabic{table}}\n")
        out.append("\\setcounter{figure}{0}\\renewcommand{\\thefigure}{S\\arabic{figure}}\n")
        out.append("\\input{supp_appendix.tex}\n\n")
        out.append(l)
        continue
    out.append(l)
open(os.path.join(AX, "main.tex"), "w").write("".join(out))

# 3) sync canonical tex sections + assets
for f in ("intro_related.tex", "method.tex", "experiment.tex", "conclusion.tex"):
    shutil.copy(os.path.join(ICRA, "tex", f), os.path.join(AX, "tex", f))
for sub in ("tables", "supp_tables", "figures"):
    src = os.path.join(ICRA, sub)
    for fn in os.listdir(src):
        if fn.endswith((".tex", ".pdf", ".png")):
            shutil.copy(os.path.join(src, fn), os.path.join(AX, sub, fn))
for f in ("references.bib", "IEEEtran.cls", "IEEEtran.bst"):
    shutil.copy(os.path.join(ICRA, f), os.path.join(AX, f))

print("make_arxiv: wrote arxiv/main.tex + arxiv/supp_appendix.tex + synced tex/assets")
