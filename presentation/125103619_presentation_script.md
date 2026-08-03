# Presentation Script  -  CS6400 (7 minutes + 3 minutes Q&A)

**Title:** EnsembleNet: Statistically Validated Demand Forecasting and Constraint-Aware Ride Pooling for Urban Shared Taxi Systems  
**Speaker:** Vishal Sharma (Student ID: 125103619)  
**Supervisor:** Dr Md Noor-A-Rahim  
**Format:** 10 slides · ~40-45 seconds per slide · total speaking time ≈ 7:00

**Delivery tips:** Face the audience, not the screen. Point to one number or figure per slide. Do not read every bullet. Pause half a second after key results (Stacking R², pool rate, city-scale impact).

---

## Slide 1  -  Title (~20 s)

Good morning / afternoon. My name is Vishal Sharma, and this is my MSc Computing Science dissertation project: EnsembleNet, on statistically validated demand forecasting and constraint-aware ride pooling for urban shared taxi systems. My supervisor is Dr Md Noor-A-Rahim.

---

## Slide 2  -  Motivation and problem (~45 s)

Cities are under pressure from congestion, cost, and emissions. Shared taxis can help, but only if two things work together.

First, operators need accurate demand forecasts so they can put vehicles where pickups will happen. Second, they need ride pooling that passengers will actually accept, not just theoretical matches.

The research gap is that these problems are usually studied separately. Forecasting papers often report a single accuracy number without testing whether model differences are statistically significant. Pooling papers often cite idealised shareability bounds, around a forty percent trip-length reduction, without hard detour and waiting constraints. My project tackles both in one pipeline on New York City Yellow Taxi data.

---

## Slide 3  -  Research questions and outline (~40 s)

I organise the work around three research questions.

RQ1: Among fourteen forecasting models, which perform best under chronological evaluation, and are the differences statistically significant?

RQ2: Does an LR-LSTM hybrid, and an XGBoost-LSTM variant, help relative to strong individual baselines?

RQ3: What pool rate and city-scale impact can a five-filter greedy pooling algorithm achieve under realistic passenger constraints?

I will cover data and method, forecasting results, pooling results, then implications and conclusions, in that order.

---

## Slide 4  -  Data and experimental design (~45 s)

I use NYC Yellow Taxi trip records. Training is January 2015, about 12.7 million trips. Testing is January to March 2016, about 34 million trips, with an eleven-month gap so evaluation is out of sample.

I build forty spatial clusters with MiniBatch KMeans, aggregate demand into ten-minute bins, and engineer lag features, cyclical time encodings, weather, and holiday indicators. All models see the same feature matrix and the same chronological split.

Ride pooling is evaluated on a sample of 106,055 trips, then scaled to the full 2015 annual volume for city-level impact.

---

## Slide 5  -  EnsembleNet models (~40 s)

The forecasting side compares fourteen models: eight individual estimators, including linear models, trees, XGBoost, gradient boosting, and an LR-LSTM hybrid; plus six ensemble strategies, including simple and weighted averaging, voting, stacking, XGBoost-LSTM, and blending.

The LR-LSTM hybrid follows Kim and colleagues: stage one fits a linear trend, stage two models residuals with an LSTM. The key design rule is fairness: identical features, identical splits, and a multi-metric evaluation for every model.

---

## Slide 6  -  Forecasting results (~50 s)

On the test set, ensembles beat every individual model on RMSE and R-squared. Stacking is best overall: RMSE 4.30, R-squared 0.9922, MAPE 10.19 percent. Blending is best on MAPE at 9.30 percent and on RMSLE.

Relative to the Kim et al. LR-LSTM benchmark of R-squared 0.572 and MAPE 15.2 percent, that is a large gain on the same dataset family, with a richer feature setup and city-wide clusters rather than a single neighbourhood.

A Friedman test on five-fold chronological cross-validation gives chi-squared 57.30 and p less than 0.001, so the rank differences are statistically significant. Practically: choose Stacking if you care about absolute dispatch error; choose Blending if you care more about percentage error.

---

## Slide 7  -  Ride pooling method (~40 s)

For pooling, I use five cascading filters before scoring a pair: pickup times within eight minutes, origins and destinations within 0.75 miles, route direction within 45 degrees, and detour at most 27 percent.

Pairs that pass all filters get a weighted compatibility score. Matching is greedy inside five-minute batches, which keeps runtime practical. This is deliberately stricter than unconstrained shareability networks, because real passengers will not accept large detours or long waits.

---

## Slide 8  -  Pooling results and impact (~50 s)

On the sample, the pool rate is 16.06 percent, with bootstrap confidence intervals over 10,000 iterations. Mean per-trip saving is about 19.7 percent, and mean compatibility is 0.66.

Compared with baselines, random pairing pools everyone but saves little; temporal-only matching pools a lot but with weak compatibility. Our method pools less, but with higher quality savings.

Scaled to roughly 146 million annual NYC Yellow Taxi trips, that projects about 28.7 million dollars in cost savings, 8.2 million miles avoided, and about 3,370 tonnes of CO2. One more finding: trip-level features alone cannot reliably predict poolability at booking time, so matching has to be reactive on concurrent trips.

---

## Slide 9  -  Implications and limitations (~40 s)

Implications: ensemble combination is more reliable than betting on one architecture; operators should optimise match quality, not raw pool rate; and pooling systems should match pairs in real time rather than pre-filter single bookings.

Limitations: weather is daily, not hourly; pooling is on a sample and pairs only; willingness to share is assumed once filters pass; detours use average speeds; and with five folds, pairwise Wilcoxon tests are underpowered, so the Friedman test is the main significance evidence.

---

## Slide 10  -  Conclusions (~30 s)

To close: I presented a unified, reproducible framework that (1) statistically validates fourteen demand models, with Stacking strongest overall; (2) evaluates residual hybrids against strong baselines; and (3) quantifies constraint-aware pooling with bootstrap confidence and city-scale impact.

Thank you. I am happy to take questions.

---

## Timing checklist (7:00)

| Block | Slides | Target |
|-------|--------|--------|
| Opening | 1-3 | 0:00-1:45 |
| Method + forecasting | 4-6 | 1:45-4:00 |
| Pooling | 7-8 | 4:00-5:40 |
| Close | 9-10 | 5:40-7:00 |

If running long, shorten Slide 5 hyperparameters talk and Slide 9 limitations to one sentence each.

---

## Likely Q&A prompts (prep notes)

**Why is your pool rate only 16% when Santi et al. say ~40%?**  
Because 40% is an unconstrained shareability bound. We enforce wait, distance, direction, and detour limits that passengers would require.

**Why Stacking over Blending?**  
Stacking wins on RMSE and R²; Blending wins on MAPE and RMSLE. It depends on the operational loss you care about.

**Is R² 0.99 realistic?**  
It is on cluster-bin pickup counts with strong lag structure. Absolute RMSE around 4 pickups per bin is the more operationally interpretable figure.

**Why not graph neural nets?**  
Road-topology graphs are not in the trip records we used; that is future work if adjacency data are available.

**Second reader / ethics / AI tools:**  
Answer factually from your thesis and CS6400 rules; do not overclaim.
