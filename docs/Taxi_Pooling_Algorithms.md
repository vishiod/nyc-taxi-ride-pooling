# Advancing Shared Taxi Systems
## Algorithm Reference — NYC Yellow Taxi Ride Pooling & Ensemble Demand Forecasting

**Author:** Vishal Sharma  
**Dataset:** NYC Yellow Taxi Trip Data (Kaggle, Jan 2015 – Mar 2016)

---

## 1. System Overview

The framework comprises five algorithmic components: (i) a ride-pooling compatibility matcher that determines whether two concurrent ride requests can share a vehicle, (ii) a pool-ability binary classifier that scores each ride's likelihood of being successfully matched, (iii) an LR-LSTM hybrid demand forecaster, (iv) an XGBoost-LSTM hybrid forecaster, and (v) six ensemble strategies that aggregate predictions from up to eight base learners. Together these constitute a 14-model comparative study evaluated on temporal holdout data from Q1 2016.

---

## 2. Ride Pooling Matcher (RidePoolingMatcher v2.0)

Given a stream of ride requests, the matcher first partitions rides into time batches of width *w* = 5 min, then applies a greedy best-match search within each batch. Compatibility is tested via five cascading Boolean filters; only pairs passing all five proceed to compatibility scoring.

---

### Algorithm 1: `is_poolable(r₁, r₂)` — Ride Compatibility Test

**Input:** Two ride records r₁, r₂ with pickup/dropoff coordinates, timestamps, trip distance.  
**Output:** `(poolable: bool, compatibility_score: float ∈ [0, 1])`

**Parameters:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| T_max | 8 min | Max temporal gap between pickups |
| D_orig | 0.75 mi | Max pickup-point separation (Haversine) |
| D_dest | 0.75 mi | Max dropoff-point separation (Haversine) |
| θ_max | 45° | Max bearing difference |
| ρ_max | 0.27 | Max detour ratio |

**Step 1 — Temporal filter:**
```
Δt = |pickup_time(r₁) − pickup_time(r₂)| / 60
if Δt > T_max  ⇒  return (False, 0.0)
```

**Step 2 — Origin proximity (Haversine):**
```
d_orig = haversine(orig(r₁), orig(r₂))
if d_orig > D_orig  ⇒  return (False, 0.0)
```

**Step 3 — Destination proximity:**
```
d_dest = haversine(dest(r₁), dest(r₂))
if d_dest > D_dest  ⇒  return (False, 0.0)
```

**Step 4 — Directional compatibility (bearing):**
```
θᵢ = atan2(sinΔλ·cosφ_dest,  cosφ_orig·sinφ_dest − sinφ_orig·cosφ_dest·cosΔλ)
Δθ = min(|θ₁ − θ₂|,  360 − |θ₁ − θ₂|)
if Δθ > θ_max  ⇒  return (False, 0.0)
```

**Step 5 — Detour ratio:**
```
d_pooled = d_orig + max(dist(r₁), dist(r₂)) + d_dest
ρ = d_pooled / (dist(r₁) + dist(r₂)) − 1.0
if ρ > ρ_max  ⇒  return (False, 0.0)
```

**Step 6 — Compatibility score (weighted sum):**
```
s_t = 1 − Δt / T_max
s_o = 1 − d_orig / D_orig
s_d = 1 − d_dest / D_dest
s_θ = 1 − Δθ / θ_max
s_ρ = 1 − ρ / ρ_max

score = 0.30·s_t + 0.25·s_o + 0.25·s_d + 0.10·s_θ + 0.10·s_ρ
return (True, score)
```

---

### Algorithm 2: `match_rides(rides, w=5 min)` — Greedy Temporal Batching

**Input:** DataFrame of ride records; time-batch width w (minutes).  
**Output:** Rides augmented with columns: `pool_id`, `pool_size`, `compatibility_score`, `estimated_savings_pct`.

```
Step 1:  Assign time batch:  batch(r) = floor(pickup_time(r) / (w × 60))
Step 2:  Initialise  pool_id ← 0;  matched ← ∅
Step 3:  For each batch Bₖ:
             For each unmatched rᵢ ∈ Bₖ:
                 best_j ← argmax_{j > i, j ∉ matched} score(rᵢ, rⱼ)
                           subject to is_poolable(rᵢ, rⱼ) = True
                 if best_j found:
                     pool_id(rᵢ) = pool_id(r_best_j) ← pool_id
                     estimated_savings(rᵢ, r_best_j) ← score × 30%
                     matched ← matched ∪ {i, best_j}
                     pool_id ← pool_id + 1
Step 4:  Rides with pool_id = −1 remain unmatched (singleton trips).
```

> **Complexity:** O(|Bₖ|²) per batch; temporal batching reduces practical runtime significantly over a global O(n²) search.

---

## 3. Pool-ability Score Predictor

A binary classifier predicts P(ride is poolable) from ride-level features. The target label is derived directly from the output of Algorithm 2: y = 1 if `pool_id(r) ≥ 0`, else y = 0.

---

### Algorithm 3: Pool-ability Feature Extraction

**Input:** Matched ride DataFrame (output of Algorithm 2).  
**Output:** Feature matrix X ∈ ℝⁿˣ⁸, label vector y ∈ {0, 1}ⁿ.

**Feature vector for ride r:**
```
φ(r) = [ trip_distance,
          trip_duration_min,
          pickup_hour,
          pickup_weekday,
          is_weekend,
          speed_mph,
          passenger_count,
          bearing(r) ]

where bearing(r) = atan2(sinΔλ·cosφ_dest,
                         cosφ_orig·sinφ_dest − sinφ_orig·cosφ_dest·cosΔλ)

Label:  y_r = 1{ pool_id(r) ≥ 0 }
```

---

## 4. LR-LSTM Hybrid Demand Forecaster

Following Kim et al. (2020), demand forecasting is decomposed into a linear component captured by ordinary least squares and a non-linear residual component modelled by an LSTM network. Features are standardised via z-score normalisation before being passed to the linear stage.

---

### Algorithm 4: LR-LSTM Hybrid — Training & Inference

**Input:** Training data (X_train, y_train); optional validation (X_val, y_val).  
**Hyperparameters:** `lstm_units=128`, `lookback L=20`, `epochs=50`, `batch_size=64`

**Stage 1 — Linear Regression:**
```
X_train_scaled ← StandardScaler().fit_transform(X_train)
β ← argmin ‖y_train − X_train_scaled · β‖²
y_hat_LR ← X_train_scaled · β
r_train  ← y_train − y_hat_LR          (residuals)
```

**Stage 2 — LSTM on residuals:**
```
Construct sequences:  {r_{t−L}, ..., r_{t−1}} → r_t   for t = L..N

LSTM architecture:
    LSTM(128, return_sequences=True) → Dropout(0.2)
    → LSTM(64) → Dropout(0.2)
    → Dense(32, ReLU) → Dense(1)

Optimiser:  Adam(η=0.001),  Loss: MSE
Callbacks:  EarlyStopping(patience=10),  ReduceLROnPlateau(factor=0.5, patience=5)
```

**Inference:**
```
y_hat_LR_test   ← X_test_scaled · β
r_hat           ← LSTM({ recent_residuals[−L:] })
y_hat_final     ← y_hat_LR_test[L:] + r_hat
```

---

## 5. XGBoost-LSTM Hybrid Demand Forecaster

An alternative two-stage hybrid that replaces the linear regression stage with gradient-boosted trees (XGBoost), allowing Stage 1 to capture non-linear feature interactions before handing residuals to the LSTM.

---

### Algorithm 5: XGBoost-LSTM Hybrid — Training & Inference

**Input:** Training data (X_train, y_train).  
**Hyperparameters:** `n_estimators=100`, `max_depth=6`, `lr=0.1`; `lstm_units=64`, `L=10`, `epochs=30`

**Stage 1 — XGBoost:**
```
XGB ← XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1)
XGB.fit(X_train, y_train)
y_hat_XGB ← XGB.predict(X_train)
r_train   ← y_train − y_hat_XGB
```

**Stage 2 — LSTM on residuals:**
```
Sequences:  {r_{t−L}, ..., r_{t−1}} → r_t

LSTM architecture:
    LSTM(64, return_sequences=True) → Dropout(0.2)
    → LSTM(32) → Dropout(0.2)
    → Dense(32, ReLU) → Dense(1)

Optimiser:  Adam(η=0.001),  Loss: MSE
```

**Inference:**
```
y_hat_XGB_test ← XGB.predict(X_test)
context        ← r_train[−L:].reshape(1, L, 1)
r_hat          ← LSTM.predict(context)[0, 0]
y_hat_final    ← y_hat_XGB_test + r_hat
```

---

## 6. Ensemble Methods

Six ensemble strategies combine predictions from M base learners {LR, Ridge, Lasso, Decision Tree, Random Forest, XGBoost, Gradient Boosting}. Methods 6A–6D are classical ensembles; 6E–6F are two-stage hybrid ensembles.

---

### Algorithm 6A: Simple Averaging Ensemble

**Input:** M trained base models {f₁, ..., f_M}, test data X_test.  
**Output:** Averaged prediction vector y_hat_SA.

```
y_hat_SA = (1/M) · Σₘ fₘ(X_test)
```

---

### Algorithm 6B: Weighted Averaging Ensemble

**Input:** M base models, validation data (X_val, y_val), test data X_test.  
**Output:** Weighted prediction y_hat_WA.

```
For each model m:
    train fₘ on X_train
    wₘ ← max(R²(fₘ, X_val, y_val), 0)

Normalise:  wₘ ← wₘ / Σₘ wₘ
y_hat_WA = Σₘ wₘ · fₘ(X_test)
```

---

### Algorithm 6C: Voting Regressor

**Input:** M base estimators, test data X_test.  
**Output:** Equal-weight averaged prediction (parallel fit via sklearn).

```
model ← VotingRegressor(estimators=[(nameₘ, fₘ) for m in 1..M])
model.fit(X_train, y_train)
y_hat_VR = model.predict(X_test)
```

---

### Algorithm 6D: Stacking Ensemble (Meta-Learning via 5-Fold CV)

**Input:** M base estimators; meta-model Ridge(α=1.0); test data X_test.  
**Output:** Meta-learned prediction y_hat_Stack.

```
model ← StackingRegressor(estimators=base_models,
                           final_estimator=Ridge(α=1.0),
                           cv=5)
model.fit(X_train, y_train)
    — internally: base models generate out-of-fold meta-features Z ∈ ℝⁿˣᴹ
    — meta-model:  α ← argmin ‖y_train − Z·α‖² + λ‖α‖²

y_hat_Stack = model.predict(X_test)
```

---

### Algorithm 6E: XGBoost-LSTM Hybrid (as Ensemble Member)

See **Algorithm 5** — the standalone XGB-LSTM model participates as an ensemble member alongside the classical base learners.

---

### Algorithm 6F: Blending Ensemble (Holdout-Based)

**Input:** M base models; meta-model Ridge(α=1.0); training data; test data X_test.  
**Output:** Blended prediction y_hat_Blend.

```
Split: (X_base, y_base), (X_meta, y_meta) ← 50/50 random split of X_train

For each model m:  fₘ.fit(X_base, y_base)

Z_meta ← column_stack( [fₘ.predict(X_meta) for m in 1..M] )   ∈ ℝ^{n/2 × M}
Ridge_meta.fit(Z_meta, y_meta)

Z_test  ← column_stack( [fₘ.predict(X_test) for m in 1..M] )
y_hat_Blend = Ridge_meta.predict(Z_test)
```

---

## 7. Evaluation Metrics

All 14 models are evaluated on the following nine metrics. SMAPE is adopted from Yin et al. (2024); Adjusted R² and MSE follow Mohammadagha (2025).

| Metric | Formula | Optimal |
|--------|---------|---------|
| RMSE | `sqrt( (1/n) · Σ(yᵢ − ŷᵢ)² )` | Lower |
| MSE | `(1/n) · Σ(yᵢ − ŷᵢ)²` | Lower |
| MAE | `(1/n) · Σ|yᵢ − ŷᵢ|` | Lower |
| MAPE | `(100/n) · Σ|yᵢ − ŷᵢ| / yᵢ` | Lower |
| SMAPE | `(100/n) · Σ|yᵢ − ŷᵢ| / ((|yᵢ| + |ŷᵢ|) / 2)` | Lower |
| R² | `1 − Σ(yᵢ−ŷᵢ)² / Σ(yᵢ−ȳ)²` | Higher |
| Adjusted R² | `1 − (1−R²)·(n−1)/(n−p−1)`,  p = num features | Higher |
| RMSLE | `sqrt( (1/n) · Σ(log(1+ŷᵢ) − log(1+yᵢ))² )` | Lower |

> **Note:** n = number of test samples; p = number of input features. Predictions are clipped to non-negative integers before metric computation (taxi demand is a count variable).

---

## 8. Base Model Hyperparameters

| Model | Key Hyperparameters |
|-------|---------------------|
| Linear Regression | OLS, no regularisation |
| Ridge Regression | alpha = 1.0 |
| Lasso Regression | alpha = 0.1 |
| Decision Tree | max_depth=10, random_state=42 |
| Random Forest | n_estimators=100, max_depth=15, max_features='sqrt', min_samples_split=5, min_samples_leaf=2 |
| XGBoost | n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8 |
| Gradient Boosting | n_estimators=150, max_depth=5, learning_rate=0.1 |
| LR-LSTM Hybrid | lstm_units=128, lookback=20, epochs=50, batch_size=64, Adam(lr=0.001) |
| XGB-LSTM Hybrid | lstm_units=64, lookback=10, epochs=30, batch_size=32, Adam(lr=0.001) |

---

*Generated from `NYC_TaxiRidePooling_AllEnsembles_2.ipynb` — all algorithms exclude data loading, pre-processing, and visualisation steps.*
