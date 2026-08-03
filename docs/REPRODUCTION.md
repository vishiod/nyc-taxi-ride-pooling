# Paper result reproduction (no re-simulation)

All frozen result data and regenerators for the EnsembleNet paper live in:

**[`nyc-taxi-ride-pooling/`](nyc-taxi-ride-pooling/)**

```bash
cd nyc-taxi-ride-pooling
pip install -r requirements-reproduce.txt
python scripts/regenerate_all.py          # all tables + figures
# or
jupyter notebook reproduce_paper_results.ipynb
```

Single artefact examples:

```bash
python scripts/regenerate_all.py --only fig:model_comp_full
python scripts/regenerate_all.py --only tab:results_full
```

- Data: [`nyc-taxi-ride-pooling/results/`](nyc-taxi-ride-pooling/results/) (see README there for file → paper label mapping)
- Paper source: [`Taxi_Clubbing (1)/main.tex`](Taxi_Clubbing%20(1)/main.tex)
- Manual architecture PDFs stay in [`Taxi_Clubbing (1)/Figures/`](Taxi_Clubbing%20(1)/Figures/) and are not rebuilt from CSV
