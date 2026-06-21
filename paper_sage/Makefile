# Makefile for paper_sage — SAGE planner research artifact.
#
# Targets:
#   make install         install pyplanner (editable) and Python deps
#   make smoke           run offline sanity tests (no LLM needed)
#   make seed            build data/memory.jsonl from eval_dataset_gt.json
#   make benchmark       run plan-quality benchmark (all default methods)
#   make benchmark-quick same but only Direct + SAGE for fast feedback
#   make ablate          run the SAGE ablation set
#   make sim             run AI2-THOR execution benchmark (requires thor_server)
#   make plots           generate figures from the latest results CSV
#   make latest          print the most recent run id
#   make clean           remove results/, figures/, data/memory.jsonl
#
# Override defaults on the command line:
#   make benchmark MODELS="llama3.2 qwen2.5:7b" METHODS="Direct SAGE"
#   make sim MODELS="llama3.2" METHODS="SAGE"

# ───────────────────────────────────────────────────────────────────────
# Paths
PAPER_DIR  := $(abspath $(CURDIR))
REPO_ROOT  := $(abspath $(PAPER_DIR)/..)
PYPLANNER  := $(REPO_ROOT)/pyplanner

# Defaults — override on command line.
PY        ?= python
OLLAMA_HOST ?= http://ollama.aistations.org
GT_PATH   ?= $(PYPLANNER)/eval_dataset_gt.json
MEMORY    ?= $(PAPER_DIR)/data/memory.jsonl

# Curated Ollama models for task planning (see README §Model selection).
MODELS    ?= llama3.2 qwen2.5:7b mistral-nemo

# Default method set (excludes LLM Router which needs external API key).
# COMMA-SEPARATED so multi-word names ("Few-Shot CoT") survive shell splitting.
# Override on CLI like: METHODS="Direct,SAGE"  or  METHODS="Few-Shot CoT".
METHODS   ?= Direct,CoT,Few-Shot CoT,Self-Refine,ReAct,Hierarchical,Hierarchical Few-Shot,SAGE

# Ablation method set (comma-separated, same convention as METHODS).
ABL_METHODS ?= SAGE,SAGE-NoVerifier,SAGE-NoRepair,SAGE-NoMemory

# Pass-through args to the benchmark driver.
EXTRA     ?=

# ───────────────────────────────────────────────────────────────────────
.PHONY: help install smoke seed benchmark benchmark-quick ablate rerun sim human-eval plots latest list-runs paper paper-verifier iccas ijcas submissions pdf pdf-iccas pdf-ijcas submit-ijcas clean-tex clean check-sota baseline-sota

help:
	@echo "paper_sage — make targets"
	@echo ""
	@echo "  install          install pyplanner (editable) + matplotlib/pandas"
	@echo "  smoke            offline sanity tests (no LLM)"
	@echo "  seed             populate data/memory.jsonl from GT plans"
	@echo "  benchmark        plan-quality benchmark, all default methods"
	@echo "  benchmark-quick  Direct + SAGE only — fast feedback loop"
	@echo "  ablate           SAGE ablation (NoVerifier / NoRepair / NoMemory)"
	@echo "  rerun            re-run one method, patch results into an existing run"
	@echo "                     usage: make rerun INTO=<run_id> METHODS=\"Few-Shot CoT\""
	@echo "                     multi: METHODS=\"Few-Shot CoT,Hierarchical Few-Shot\""
	@echo "                            (comma-separated, spaces preserved)"
	@echo "  sim              AI2-THOR execution benchmark (needs thor_server.py)"
	@echo "  human-eval       offline: re-score + reconcile the shipped 72-item"
	@echo "                     end-task bundle (no LLM/sim); reproduces 0.53/0.40/0.28"
	@echo "  plots            figures from results.csv (latest by default)"
	@echo "                     override: make plots RUN_ID=<id>            pick a run"
	@echo "                               make plots LIST=1                 list runs"
	@echo "                               make plots EXTRA_CSV=<path>...    merge extras"
	@echo "  latest           print the most recent run id"
	@echo "  list-runs        list all runs with row counts"
	@echo "  paper            refresh paper/ with completeness-variant abstract"
	@echo "                     (calls init_paper.py: abstract + bib + figures)"
	@echo "  paper-verifier   same but with the verifier-variant abstract"
	@echo "  iccas            emit ICCAS/ short version (1-2 page extended abstract)"
	@echo "  ijcas            emit IJCAS/ full version (6+ page journal article)"
	@echo "  submissions      emit BOTH ICCAS/ and IJCAS/ (typical pre-submission)"
	@echo "                     override: VARIANT=verifier to use that abstract"
	@echo "                               RUN_ID_PAPER=<id> to pin figure run"
	@echo "  pdf              compile ICCAS/main.pdf AND IJCAS/main.pdf"
	@echo "  pdf-iccas        compile only ICCAS/main.pdf"
	@echo "  pdf-ijcas        compile only IJCAS/main.pdf"
	@echo "  submit-ijcas     zip the minimal IJCAS source set for submission"
	@echo "                     output: $(notdir $(SUBMIT_ZIP)) (override SUBMIT_ZIP=<path>)"
	@echo "  clean-tex        remove LaTeX aux files (.aux, .log, .out, .bbl, ...)"
	@echo "  clean            remove results/, figures/, data/memory.jsonl"
	@echo ""
	@echo "Variables (override on CLI):"
	@echo "  MODELS=\"llama3.2 qwen2.5:7b\""
	@echo "  METHODS=\"Direct SAGE\""
	@echo "  OLLAMA_HOST=http://localhost:11434"

install:
	@echo "→ installing pyplanner (editable)…"
	$(PY) -m pip install -e $(PYPLANNER)
	@echo "→ installing benchmark deps…"
	$(PY) -m pip install pandas matplotlib chromadb requests
	@echo "→ installing SOTA-baseline dep (pyperplan, for LLM+P)…"
	$(PY) -m pip install pyperplan
	@echo "→ verifying SAGE import…"
	@$(PY) -c "import pyplanner; assert 'SAGE' in pyplanner.REGISTRY; print('  ok — SAGE registered')"

smoke:
	$(PY) scripts/smoke_test.py

# ── published-style SOTA baselines (LLM+P, SayCan) ────────────────────
# Offline acceptance gate for the SOTA baselines (no LLM). Requires the
# pure-python classical planner pyperplan:
#   /home/keti/miniconda3/bin/pip install pyperplan
SOTA_METHODS ?= Direct,SAGE,LLM+P,SayCan

check-sota:
	$(PY) scripts/check_sota.py

# Run the SOTA comparison set (Direct + SAGE + LLM+P + SayCan) on one model.
baseline-sota: seed
	$(PY) scripts/run_benchmark.py \
		--mode plan \
		--host  $(OLLAMA_HOST) \
		--models $(firstword $(MODELS)) \
		--methods-csv "$(SOTA_METHODS)" \
		--gt-path $(GT_PATH) \
		--live-path $(MEMORY) \
		$(EXTRA)

seed: $(MEMORY)

$(MEMORY):
	@mkdir -p $(dir $(MEMORY))
	$(PY) scripts/seed_memory.py --gt $(GT_PATH) --out $(MEMORY)

benchmark: seed
	$(PY) scripts/run_benchmark.py \
		--mode plan \
		--host  $(OLLAMA_HOST) \
		--models $(MODELS) \
		--methods-csv "$(METHODS)" \
		--gt-path $(GT_PATH) \
		--live-path $(MEMORY) \
		$(EXTRA)

benchmark-quick: seed
	$(PY) scripts/run_benchmark.py \
		--mode plan \
		--host  $(OLLAMA_HOST) \
		--models $(firstword $(MODELS)) \
		--methods-csv "Direct,SAGE" \
		--gt-path $(GT_PATH) \
		--live-path $(MEMORY) \
		--max-samples 5 \
		$(EXTRA)

ablate: seed
	$(PY) scripts/run_benchmark.py \
		--mode plan \
		--host  $(OLLAMA_HOST) \
		--models $(MODELS) \
		--methods-csv "$(ABL_METHODS)" \
		--gt-path $(GT_PATH) \
		--live-path $(MEMORY) \
		$(EXTRA)

# ── rerun ─────────────────────────────────────────────────────────────
# Re-run a specific method (and optional model subset) and patch the
# results directly back into an existing run's CSV.
#
# Required: INTO=<run_id>   target run that already has results.csv
# Required: METHODS="..."   space-separated method name(s) to rerun
# Optional: MODELS="..."    models to rerun (defaults to MODELS above)
#
# Example — re-run Few-Shot CoT on 3 models, patch into the May-19 run:
#   make rerun INTO=20260519_143552_c7c7ec METHODS="Few-Shot CoT"
#
# Example — re-run just on llama3.2:
#   make rerun INTO=20260519_143552_c7c7ec METHODS="Few-Shot CoT" MODELS=llama3.2
#
# The existing CSV is backed up to results.csv.bak.<ts> before patching.
INTO ?=

rerun: seed
	@if [ -z "$(INTO)" ]; then \
		echo "✗ INTO=<run_id> is required. Use 'make list-runs' to see options."; \
		exit 1; \
	fi; \
	if [ ! -d "results/$(INTO)" ]; then \
		echo "✗ run not found: results/$(INTO)/"; exit 1; \
	fi
	$(PY) scripts/run_benchmark.py \
		--mode plan \
		--host  $(OLLAMA_HOST) \
		--models $(MODELS) \
		--methods-csv "$(METHODS)" \
		--gt-path $(GT_PATH) \
		--live-path $(MEMORY) \
		--into $(INTO) \
		$(EXTRA)

sim: seed
	@echo "→ make sure 'python thor_server.py' is running in another terminal."
	$(PY) scripts/run_benchmark.py \
		--mode sim \
		--host $(OLLAMA_HOST) \
		--models $(MODELS) \
		--methods-csv "$(METHODS)" \
		--gt-path $(GT_PATH) \
		--live-path $(MEMORY) \
		$(EXTRA)

# ── plotting ──────────────────────────────────────────────────────────
# By default plots the most recent run. Override with RUN_ID:
#   make plots                          → latest
#   make plots RUN_ID=20260519_143003_792d7d   → specific run
#   make plots RUN_ID=latest            → same as default, explicit
#   make plots LIST=1                   → list runs and exit
RUN_ID    ?= latest
LIST      ?= 0
# Space-separated list of additional CSV paths whose (model, method) rows
# REPLACE the matching rows in the base CSV at plot time. Use this when
# you re-ran one method into a new run and want the figure to show the
# fresh data alongside the unchanged rows of the original run.
EXTRA_CSV ?=

latest:
	@ls -td results/*/ 2>/dev/null | head -n 1 || (echo "no runs yet"; exit 1)

list-runs:
	@if [ ! -d results ] || [ -z "$$(ls -A results 2>/dev/null)" ]; then \
		echo "no runs yet — run 'make benchmark' first"; exit 1; \
	fi; \
	echo "available runs (newest first):"; \
	for d in $$(ls -td results/*/); do \
		id=$$(basename $$d); \
		csv="$$d/results.csv"; \
		if [ -f "$$csv" ]; then \
			n=$$(($$( wc -l < "$$csv" ) - 1)); \
			echo "  $$id  ($$n rows)"; \
		else \
			echo "  $$id  (no results.csv)"; \
		fi; \
	done

plots:
	@if [ "$(LIST)" = "1" ]; then $(MAKE) -s list-runs; exit 0; fi; \
	if [ "$(RUN_ID)" = "latest" ] || [ -z "$(RUN_ID)" ]; then \
		TARGET=$$(ls -td results/*/ 2>/dev/null | head -n 1); \
	else \
		TARGET="results/$(RUN_ID)/"; \
	fi; \
	if [ -z "$$TARGET" ] || [ ! -d "$$TARGET" ]; then \
		echo "✗ run not found: $${TARGET:-<empty>}"; \
		echo "  available runs:"; $(MAKE) -s list-runs 2>/dev/null | tail -n +2; \
		exit 1; \
	fi; \
	if [ ! -f "$$TARGET/results.csv" ]; then \
		echo "✗ no results.csv in $$TARGET"; exit 1; \
	fi; \
	if [ -n "$(EXTRA_CSV)" ]; then \
		echo "→ plotting $$TARGET/results.csv (with extras: $(EXTRA_CSV))"; \
		$(PY) scripts/plot_results.py "$$TARGET/results.csv" --extra $(EXTRA_CSV); \
	else \
		echo "→ plotting $$TARGET/results.csv"; \
		$(PY) scripts/plot_results.py "$$TARGET/results.csv"; \
	fi

# ── human-judged end-task success ─────────────────────────────────────
# Offline (no LLM, no simulator): re-score the shipped 72-item bundle with
# the partial-credit auto-checker, then reconcile against the rater CSVs.
# Reproduces the paper's end-task numbers (SAGE 0.53 > Direct 0.40 > Hier 0.28,
# Cohen's kappa 0.65, auto-vs-human Pearson r 0.82). Bundle lives at
# results/human_eval/ (frames + items.jsonl + rater CSVs). To re-capture a
# fresh bundle from scratch see dataset_release/REPRODUCE.md §6.
HE_BUNDLE ?= results/human_eval
human-eval:
	@if [ ! -f "$(HE_BUNDLE)/items.jsonl" ]; then \
		echo "✗ no bundle at $(HE_BUNDLE)/ — see REPRODUCE.md §6 to capture one"; exit 1; \
	fi
	$(PY) scripts/auto_partial_score.py
	$(PY) scripts/reconcile_auto_human.py
	@echo "→ open $(HE_BUNDLE)/judge.html to inspect the blinded rater interface"

# ── paper assembly ────────────────────────────────────────────────────
# Refresh paper/ artifacts from the latest figures + literature_survey.md.
#   make paper              → completeness-variant abstract (default)
#   make paper-verifier     → verifier-variant abstract
#   make paper RUN_ID=...   → pin figures to a specific run
RUN_ID_PAPER ?= latest
VARIANT      ?= completeness

paper:
	$(PY) scripts/init_paper.py --target paper --variant $(VARIANT) --run-id $(RUN_ID_PAPER)

paper-verifier:
	$(PY) scripts/init_paper.py --target paper --variant verifier --run-id $(RUN_ID_PAPER)

# ── dual submission (ICCAS 2026 + IJCAS Special Issue) ────────────────
# See SUBMISSION_PLAN.md for the joint-submission workflow. The two
# targets share source-of-truth in paper/ but emit to different folders
# with venue-specific templates and bibliography formats.
#   make iccas              → ICCAS/ (1-2 page extended abstract)
#   make ijcas              → IJCAS/ (6+ page full journal version)
#   make submissions        → both, in one shot
#   make submissions VARIANT=verifier  → use the verifier-first abstract
iccas:
	$(PY) scripts/init_paper.py --target iccas --variant $(VARIANT) --run-id $(RUN_ID_PAPER)

ijcas:
	$(PY) scripts/init_paper.py --target ijcas --variant $(VARIANT) --run-id $(RUN_ID_PAPER)

submissions:
	$(PY) scripts/init_paper.py --target both --variant $(VARIANT) --run-id $(RUN_ID_PAPER)

# ── PDF compilation ──────────────────────────────────────────────────
# Compile the .tex files inside ICCAS/ and IJCAS/ into PDFs. The targets
# do NOT depend on `submissions`, so hand-edits to ICCAS/ or IJCAS/ are
# never overwritten on `make pdf`. Run `make submissions && make pdf`
# (or `make submissions pdf`) when you want a fresh content refresh.
#
# Uses latexmk -pdf if available (handles re-runs automatically); falls
# back to two pdflatex passes otherwise. Aux files are intentionally
# left in place for fast incremental compiles; clean them with
# `make clean-tex`.
#
# Required tools (install once on Ubuntu/Debian):
#   sudo apt install -y texlive-latex-base texlive-latex-recommended \
#                       texlive-latex-extra texlive-fonts-recommended \
#                       texlive-science latexmk
LATEXMK := $(shell command -v latexmk 2>/dev/null)
PDFLATEX := $(shell command -v pdflatex 2>/dev/null)

define compile_pdf
	@if [ -z "$(PDFLATEX)" ]; then \
		echo "✗ pdflatex not found. Install with:"; \
		echo "    sudo apt install -y texlive-latex-base texlive-latex-recommended \\"; \
		echo "                        texlive-latex-extra texlive-fonts-recommended \\"; \
		echo "                        texlive-science latexmk"; \
		exit 1; \
	fi
	@if [ ! -f "$(1)/main.tex" ]; then \
		echo "✗ $(1)/main.tex not found. Run 'make $(2)' first."; \
		exit 1; \
	fi
	@echo "→ compiling $(1)/main.tex"
	@if [ -n "$(LATEXMK)" ]; then \
		cd $(1) && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex; \
	else \
		cd $(1) && pdflatex -interaction=nonstopmode -halt-on-error main.tex \
		      && pdflatex -interaction=nonstopmode -halt-on-error main.tex; \
	fi
	@echo "  ✓ $(1)/main.pdf ($$( du -h $(1)/main.pdf | cut -f1 ))"
endef

pdf-iccas:
	$(call compile_pdf,ICCAS,iccas)

pdf-ijcas:
	$(call compile_pdf,IJCAS,ijcas)

pdf: pdf-iccas pdf-ijcas
	@echo ""
	@echo "Both PDFs ready:"
	@echo "  ICCAS/main.pdf  (extended abstract — 1-2 pages)"
	@echo "  IJCAS/main.pdf  (full journal version — 6+ pages)"

# ── submission packaging ─────────────────────────────────────────────
# Bundle the *minimal* set of files IJCAS needs to compile, with every
# symlink dereferenced into a real file. Only the figures/tables/portraits
# actually referenced by the .tex sources are pulled in, and figures/
# portraits only in their .pdf form (the build uses pdflatex). No main.pdf,
# no aux/build artifacts. The IJCAS/ folder is packaged as-is — run
# `make ijcas && make pdf-ijcas` first if you want it refreshed.
#
# NOTE: the recipe is one backslash-continued shell command so $$stage
# persists across steps — that's why there are no `#` comments inside it.
IJCAS_DIR  ?= $(PAPER_DIR)/IJCAS
SUBMIT_ZIP ?= $(PAPER_DIR)/ijcas_submission.zip

submit-ijcas:
	@if [ ! -f "$(IJCAS_DIR)/main.tex" ]; then \
		echo "✗ $(IJCAS_DIR)/main.tex not found. Run 'make ijcas' first."; \
		exit 1; \
	fi
	@stage=$$(mktemp -d); \
	echo "→ staging minimal IJCAS submission"; \
	cd "$(IJCAS_DIR)"; \
	mkdir -p "$$stage/tex" "$$stage/figs" "$$stage/tables" "$$stage/portraits"; \
	for f in main.tex references.bib IJCAS.cls picins.sty; do \
		if [ -f "$$f" ]; then cp -L "$$f" "$$stage/"; \
		else echo "  [warn] missing $$f"; fi; \
	done; \
	cp -L tex/*.tex "$$stage/tex/" 2>/dev/null || true; \
	for fig in $$(grep -hoE 'includegraphics(\[[^]]*\])?\{figs/[^}]*\}' main.tex tex/*.tex | sed -E 's/.*\{figs\/([^}]*)\}/\1/' | sort -u); do \
		if [ -f "figs/$$fig" ]; then cp -L "figs/$$fig" "$$stage/figs/"; \
		else echo "  [warn] referenced figure missing: figs/$$fig"; fi; \
	done; \
	for tbl in $$(grep -hoE '\\input\{tables/[^}]*\}' main.tex tex/*.tex | sed -E 's/.*\{tables\/([^}]*)\}/\1/' | sort -u); do \
		if [ -f "tables/$$tbl.tex" ]; then cp -L "tables/$$tbl.tex" "$$stage/tables/"; \
		elif [ -f "tables/$$tbl" ]; then cp -L "tables/$$tbl" "$$stage/tables/"; \
		else echo "  [warn] referenced table missing: tables/$$tbl"; fi; \
	done; \
	for por in $$(grep -hoE 'biography\{portraits/[^}]*\}' tex/*.tex | sed -E 's/.*\{portraits\/([^}]*)\}/\1/' | sort -u); do \
		if [ -f "portraits/$$por.pdf" ]; then cp -L "portraits/$$por.pdf" "$$stage/portraits/"; \
		else echo "  [warn] referenced portrait missing: portraits/$$por.pdf"; fi; \
	done; \
	rmdir "$$stage/figs" "$$stage/tables" "$$stage/portraits" 2>/dev/null || true; \
	rm -f "$(SUBMIT_ZIP)"; \
	( cd "$$stage" && zip -r -X "$(SUBMIT_ZIP)" . >/dev/null ); \
	echo ""; \
	echo "✓ $(SUBMIT_ZIP) ($$( du -h "$(SUBMIT_ZIP)" | cut -f1 ))"; \
	echo "  contents:"; \
	( cd "$$stage" && find . -type f | sed 's|^\./|    |' | sort ); \
	rm -rf "$$stage"

clean-tex:
	@for d in paper ICCAS IJCAS; do \
		if [ -d $$d ]; then \
			echo "→ cleaning $$d/ aux files"; \
			rm -f $$d/main.aux $$d/main.log $$d/main.out $$d/main.bbl \
			      $$d/main.blg $$d/main.toc $$d/main.lof $$d/main.lot \
			      $$d/main.fls $$d/main.fdb_latexmk $$d/main.synctex.gz; \
			rm -f $$d/tex/*.aux 2>/dev/null || true; \
		fi; \
	done
	@echo "✓ aux files removed (main.pdf left in place)"

clean:
	@# preserve results/human_eval/ — it is a shipped artifact, not a reproducible run
	find results -mindepth 1 -maxdepth 1 ! -name human_eval -exec rm -rf {} + 2>/dev/null || true
	rm -rf figures data/memory.jsonl
	@echo "cleaned results/* (kept human_eval/), figures/, data/memory.jsonl"
