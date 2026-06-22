# License & Attribution Notes

This release bundles three kinds of material with different recommended
licenses, plus a third-party dependency (AI2-THOR) with its own terms. Read
this before redistributing.

---

## 1. Recommended licensing for this release

**Dataset (the task records: instructions, reference plans, metadata).**
Recommend **CC-BY-4.0** (Creative Commons Attribution 4.0). This permits reuse
and redistribution with attribution, which suits a benchmark intended for
community comparison.

- Files: `eval_dataset_gt.json`, `eval_dataset_expanded.json`,
  `eval_dataset_procthor.json`, and the COMPOUND output.

**Code (generators, benchmark driver, verifier, planner).**
Recommend **MIT** or **Apache-2.0** (Apache-2.0 if an explicit patent grant is
desired). Either is permissive and standard for research tooling.

- Files: `make_dataset.py`, `expand_dataset.py`, `gen_compound_tasks.py`,
  `build_procthor_dataset.py`, `run_benchmark.py`, `run_grid.sh`,
  `pyplanner/` modules (`sage.py`, `verifier.py`, `memory_retriever.py`, etc.).

Add a top-level `LICENSE` (code) and a `DATA_LICENSE` (or a license note inside
the dataset directory) so the two are unambiguous.

---

## 2. AI2-THOR (third-party) — required attribution

The simulator and **all 3D scenes / FloorPlans are AI2-THOR**, developed by the
**Allen Institute for AI (AI2)**. This release does **not** relicense AI2-THOR
or its assets.

- AI2-THOR is distributed under its **own license** (Apache-2.0 for the
  framework; scene assets are AI2-THOR's). Comply with AI2-THOR's license terms
  and attribution requirements when redistributing anything derived from it.
- Our dataset records *reference* AI2-THOR scenes (e.g. `FloorPlan1`) and object
  types; the scenes themselves are not redistributed here and remain AI2-THOR's.
- **Cite AI2-THOR** (Kolve et al., *AI2-THOR: An Interactive 3D Environment for
  Visual AI*) and credit the Allen Institute for AI in any publication or
  redistribution.
- The CC-BY-4.0 recommendation above covers **our task annotations only**, not
  AI2-THOR assets.

If the release includes ProcTHOR-derived data, the same applies to **ProcTHOR**
(also from AI2): cite it and follow its license/attribution.

---

## 3. Anonymization for double-blind review

For double-blind venues, before release of the artifact or any supplementary
material:

- Remove author names, institutional affiliations, internal usernames, and
  absolute home paths (e.g. `/home/<user>/...`, `/media/<user>/...`) from
  scripts, configs, logs, and docs.
- Strip identifying hostnames from defaults (e.g. the Ollama host
  `localhost:11434`) or replace with a neutral placeholder.
- Scrub `grounded_at` timestamps only if they could de-anonymize; otherwise they
  are harmless provenance.
- Host the anonymized artifact on an anonymous repository (e.g. an
  anonymized-link service) and reference it from the paper without revealing
  author identity.
- Restore real author/license/attribution information in the camera-ready
  release.

---

## 4. Summary

| Component | Recommended license | Notes |
|-----------|---------------------|-------|
| Task annotations (dataset JSON) | CC-BY-4.0 | Our contribution; attribute on reuse. |
| Code (generators, driver, planner, verifier) | MIT or Apache-2.0 | Apache-2.0 if patent grant wanted. |
| AI2-THOR framework + scenes | AI2-THOR's own license | Allen Institute for AI; cite and attribute. |
| ProcTHOR (if included) | ProcTHOR's own license | Allen Institute for AI; cite and attribute. |
