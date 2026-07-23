# PACT Track 2 IJCAI technical report

This directory contains the report for the final PACT obligation runtime. The
CAR-bench limit is up to four pages of main text; references are excluded from
that limit. The report describes the submitted PACT V2 path only; earlier
selective workflow prototypes and their development scores are not evidence for
this system.

The IJCAI-ECAI 2026 author-kit files are vendored unchanged under
`official_template/`. Build from this directory with:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The expected outputs are `main.pdf` and
`PACT_CAR-bench_Track2_Technical_Report.pdf`. Verify the deliverable before
uploading:

```bash
pdfinfo main.pdf | grep '^Pages:'
grep -E 'Overfull|Underfull|undefined|Warning' main.log
```

The report deliberately makes no hidden-set or V2 task-score claim. Its test
counts are a dated engineering snapshot and should be updated if the final
release adds tests. The immutable GHCR digest belongs in the submission form
after the candidate image is built and validated; no provisional digest is
embedded in the report.
