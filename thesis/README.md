# UCC MSc Thesis (uccthesis)

LaTeX source for Vishal Sharma's MSc Computing Science thesis, converted from the EnsembleNet paper into the official UCC `uccthesis` class (from Canvas `Thesis.zip`). Structure follows the example thesis shared by the supervisor (`123113268_Ninglin_Ou.pdf`).

## Compile (Overleaf recommended)

1. Upload the entire `thesis/` folder to Overleaf.
2. Set compiler to **pdfLaTeX**.
3. Set bibliography tool to **Biber** (not BibTeX).
4. Main file: `main.tex`.
5. Recompile twice after Biber.

Locally (TeX Live / MacTeX):

```bash
cd thesis
pdflatex main
biber main
pdflatex main
pdflatex main
```

## Before submission

- Replace `\secondreader{TBD}` in `main.tex` with your assigned second reader.
- Confirm the date on the title page.
- Rename the Canvas abstract file `Canvas things/STUDENT_ID.txt` to your real student ID (e.g. `123456789.txt`).
