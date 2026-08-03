# Advancing Shared Taxi Systems — MSc Dissertation Project

Author: Vishal Sharma (UCC Student ID: 125103619)

This repository contains the full project: source code/notebooks, LaTeX thesis and paper sources, figures, and reproduced results.

## Layout

- `NYC_TaxiRidePooling_AllEnsembles.ipynb` — main research notebook (ensembles + ride pooling)
- `reproduce_paper_results.ipynb` / `scripts/` — reproduction helpers
- `results/`, `reproduced_figures/`, `reproduced_tables/` — tables/figures used in writing
- `thesis/` — UCC thesis LaTeX sources and figures
- `paper/` — CAS-format paper LaTeX sources and figures
- `docs/` — supporting notes
- `requirements.txt` — Python dependencies

## Build thesis

```bash
cd thesis
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Run notebook

```bash
pip install -r requirements.txt
jupyter notebook NYC_TaxiRidePooling_AllEnsembles.ipynb
```

Requires a Kaggle API token for NYC Yellow Taxi data download.
