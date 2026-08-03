# EnsembleNet — CS6400 Dissertation Project

**Author:** Vishal Sharma  
**Student ID:** 125103619  
**Programme:** MSc Computing Science (CS6400)  
**Supervisor:** Dr Md Noor-A-Rahim  

Unified project repository for the CS6400 dissertation: thesis LaTeX + figures, implementation source, and derived results. Official supervisor sharing is via the **pre-configured bare Git repository on `csgate.ucc.ie`** (in progress with CS IT). This GitHub copy is a working mirror until csgate access is confirmed.

## CS6400 software / repo requirements (tracked here)

| Requirement | How this repo addresses it |
|-------------|----------------------------|
| Thesis LaTeX (`uccthesis`) + graphics | `thesis/` |
| Source code | `*.ipynb`, `scripts/` |
| Incremental history on `master` | `master` branch (mirrors development history) |
| No large raw data files | Raw taxi CSVs excluded; download via Kaggle |
| Freeze after electronic thesis submission | Do not push after Canvas thesis upload |

**Hard deadline (provisional):** software + electronic thesis — 23:59, 30 August 2026.  
**Thesis PDF name on Canvas:** `125103619.pdf`  
Presentation deliverables are separate and not covered here yet.

## Layout

```
thesis/                         # uccthesis sources (compile to 125103619.pdf)
  chapters/                     # Chs 1–7 + appendix
  Figures/                      # thesis figures (PDF)
NYC_TaxiRidePooling_AllEnsembles.ipynb
reproduce_paper_results.ipynb
scripts/                        # regeneration helpers
results/                        # derived metrics (small CSVs/JSON)
reproduced_tables/              # LaTeX table fragments
reproduced_figures/             # exported figure PDFs/PNGs
paper/                          # companion CAS paper sources
docs/                           # notes / compliance
requirements.txt
```

## Build thesis

```bash
cd thesis
pdflatex main
biber main
pdflatex main
pdflatex main
# submit export as 125103619.pdf
```

## Run code

```bash
pip install -r requirements.txt
jupyter notebook NYC_TaxiRidePooling_AllEnsembles.ipynb
```

Requires a Kaggle API token for NYC Yellow Taxi data (not stored in this repo).

## Branching

- **`master`** — canonical branch for CS6400 (full project history).
- `main` may exist as an alias during transition; prefer `master` for supervisor sharing on csgate.
