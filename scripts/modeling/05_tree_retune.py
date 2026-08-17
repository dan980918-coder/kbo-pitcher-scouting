#!/usr/bin/env python3
"""
Modeling step 4b: RF/GB retuned on a narrow grid (centered on the setting
that worked when tried plain), with repeated 3-fold CV (3-fold x 5 seeds =
15 fold evaluations) to separate stable configs from lucky ones.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RepeatedKFold, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import itertools

ROOT = "/Users/ojaemin/Downloads/kbo-pitcher-scouting"
df = pd.read_csv(f"{ROOT}/data/rosters/analysis_dataset_v1.csv")
TARGET = "kbo_first_year_WAR"

modeling_pop = df[df["kbo_no_appearance"] != 1].copy()
train = modeling_pop[modeling_pop["연도"] <= 2023].copy()
val = modeling_pop[modeling_pop["연도"] == 2024].copy()

LEVEL_FEATURES = {"mlb": ["mlb_fip_last", "mlb_fip_minus_career"], "aaa": ["aaa_hr9_last", "aaa_bb9_3yr"]}
OTHER_FEATURES = ["age_at_kbo_entry", "n_pitch_types_recorded"]
ALL_RAW = LEVEL_FEATURES["mlb"] + LEVEL_FEATURES["aaa"] + OTHER_FEATURES


def build(d, tm, tom):
    d = d.copy()
    d["has_mlb_record"] = (d["mlb_career_ip"].fillna(0) > 0).astype(int)
    d["has_aaa_record"] = (d["aaa_career_ip"].fillna(0) > 0).astype(int)
    for level, cols in LEVEL_FEATURES.items():
        hc = f"has_{level}_record"
        for col in cols:
            d[col] = d[col].where(d[hc] == 1, tm[col])
            d[col] = d[col].fillna(tm[col])
    for col in OTHER_FEATURES:
        d[col] = d[col].fillna(tom[col])
    return d


train["has_mlb_record_tmp"] = (train["mlb_career_ip"].fillna(0) > 0).astype(int)
train["has_aaa_record_tmp"] = (train["aaa_career_ip"].fillna(0) > 0).astype(int)
tm = {}
for level, cols in LEVEL_FEATURES.items():
    hc = f"has_{level}_record_tmp"
    for col in cols:
        tm[col] = train.loc[train[hc] == 1, col].mean()
tom = {col: train[col].mean() for col in OTHER_FEATURES}

train_f = build(train, tm, tom)
val_f = build(val, tm, tom)
FEATURE_COLS = ALL_RAW + ["has_mlb_record", "has_aaa_record"]
train_fit = train_f.dropna(subset=[TARGET])
val_fit = val_f.dropna(subset=[TARGET])

X_train = train_fit[FEATURE_COLS].values
y_train = train_fit[TARGET].values
X_val = val_fit[FEATURE_COLS].values
y_val = val_fit[TARGET].values

rkf = RepeatedKFold(n_splits=3, n_repeats=5, random_state=42)  # 15 fold evaluations


def metrics(y_true, y_pred):
    return {"MAE": round(mean_absolute_error(y_true, y_pred), 3),
            "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 3),
            "R2": round(r2_score(y_true, y_pred), 3)}


# ---- 1-2. narrow grid, repeated CV ----
grid = {"max_depth": [2, 3], "n_estimators": [20, 50, 100], "min_samples_leaf": [10, 15, 20]}
grid_combos = list(itertools.product(grid["max_depth"], grid["n_estimators"], grid["min_samples_leaf"]))
gb_learning_rates = [0.05, 0.1]  # not specified by user; GB needs one, kept narrow/conservative

print("=" * 90)
print("1-2. RF/GB 좁힌 그리드 x Repeated 3-fold CV (3-fold x 5 seeds = 15회 평가)")
print("=" * 90)

rf_rows = []
for depth, n_est, leaf in grid_combos:
    model = RandomForestRegressor(max_depth=depth, n_estimators=n_est, min_samples_leaf=leaf, random_state=42)
    mae_scores = -cross_val_score(model, X_train, y_train, cv=rkf, scoring="neg_mean_absolute_error", n_jobs=1)
    r2_scores = cross_val_score(model, X_train, y_train, cv=rkf, scoring="r2", n_jobs=1)
    rf_rows.append({"max_depth": depth, "n_estimators": n_est, "min_samples_leaf": leaf,
                     "cv_MAE_mean": mae_scores.mean(), "cv_MAE_std": mae_scores.std(),
                     "cv_R2_mean": r2_scores.mean(), "cv_R2_std": r2_scores.std()})
rf_table = pd.DataFrame(rf_rows).sort_values("cv_MAE_mean")
print("\n--- Random Forest (상위 10개, cv_MAE_mean 기준) ---")
print(rf_table.head(10).round(4).to_string(index=False))

gb_rows = []
for depth, n_est, leaf in grid_combos:
    for lr in gb_learning_rates:
        model = GradientBoostingRegressor(max_depth=depth, n_estimators=n_est, min_samples_leaf=leaf,
                                           learning_rate=lr, random_state=42)
        mae_scores = -cross_val_score(model, X_train, y_train, cv=rkf, scoring="neg_mean_absolute_error", n_jobs=1)
        r2_scores = cross_val_score(model, X_train, y_train, cv=rkf, scoring="r2", n_jobs=1)
        gb_rows.append({"max_depth": depth, "n_estimators": n_est, "min_samples_leaf": leaf, "learning_rate": lr,
                         "cv_MAE_mean": mae_scores.mean(), "cv_MAE_std": mae_scores.std(),
                         "cv_R2_mean": r2_scores.mean(), "cv_R2_std": r2_scores.std()})
gb_table = pd.DataFrame(gb_rows).sort_values("cv_MAE_mean")
print("\n--- Gradient Boosting (상위 10개, cv_MAE_mean 기준) ---")
print(gb_table.head(10).round(4).to_string(index=False))

# ---- 3. stability-aware selection: among top 5 by mean, pick lowest std ----
def pick_stable(table, top_n=5):
    top = table.head(top_n).copy()
    return top.loc[top["cv_MAE_std"].idxmin()]


rf_pick = pick_stable(rf_table)
gb_pick = pick_stable(gb_table)
print("\n" + "=" * 90)
print("3. 안정성 고려 선택 (상위 5개 중 std 가장 낮은 조합)")
print("=" * 90)
print("RF 선택:", rf_pick.to_dict())
print("GB 선택:", gb_pick.to_dict())

# ---- 4. final val evaluation, all models together ----
rf_final = RandomForestRegressor(max_depth=int(rf_pick["max_depth"]), n_estimators=int(rf_pick["n_estimators"]),
                                  min_samples_leaf=int(rf_pick["min_samples_leaf"]), random_state=42)
gb_final = GradientBoostingRegressor(max_depth=int(gb_pick["max_depth"]), n_estimators=int(gb_pick["n_estimators"]),
                                      min_samples_leaf=int(gb_pick["min_samples_leaf"]),
                                      learning_rate=gb_pick["learning_rate"], random_state=42)
rf_final.fit(X_train, y_train)
gb_final.fit(X_train, y_train)

pred_val_rf = rf_final.predict(X_val)
pred_train_rf = rf_final.predict(X_train)
pred_val_gb = gb_final.predict(X_val)
pred_train_gb = gb_final.predict(X_train)

# baseline
def baseline_feature(d):
    return d["mlb_fip_last"].where(d["mlb_fip_last"].notna(), d["aaa_fip_last"])


train["baseline_fip"] = baseline_feature(train)
val_fit["baseline_fip"] = baseline_feature(val_fit)
train_bl_fit = train.dropna(subset=["baseline_fip", TARGET])
baseline_model = LinearRegression().fit(train_bl_fit[["baseline_fip"]].values, train_bl_fit[TARGET].values)
pred_val_bl = np.where(val_fit["baseline_fip"].notna(),
                        baseline_model.predict(val_fit[["baseline_fip"]].fillna(0).values),
                        train_bl_fit[TARGET].mean())
pred_train_bl = np.where(train["baseline_fip"].notna(),
                          baseline_model.predict(train[["baseline_fip"]].fillna(0).values),
                          train_bl_fit[TARGET].mean())

# ridge
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
alphas = np.logspace(-2, 3, 60)
cv5 = KFold(n_splits=5, shuffle=True, random_state=42)
ridge = RidgeCV(alphas=alphas, cv=cv5, scoring="neg_mean_absolute_error")
ridge.fit(X_train_s, y_train)
pred_val_ridge = ridge.predict(X_val_s)
pred_train_ridge = ridge.predict(X_train_s)

# extreme-simple RF from before, for reference row
rf_simple = RandomForestRegressor(max_depth=2, n_estimators=20, min_samples_leaf=10, random_state=42)
rf_simple.fit(X_train, y_train)
pred_val_simple = rf_simple.predict(X_val)
pred_train_simple = rf_simple.predict(X_train)
mae_simple_cv = -cross_val_score(rf_simple, X_train, y_train, cv=rkf, scoring="neg_mean_absolute_error", n_jobs=1)
r2_simple_cv = cross_val_score(rf_simple, X_train, y_train, cv=rkf, scoring="r2", n_jobs=1)

print("\n" + "=" * 90)
print("4. 전체 모델 비교 -- CV 성능 vs Validation(2024, n=18) 성능")
print("=" * 90)
rows = []
rows.append({"model": "Baseline", "cv_MAE": np.nan, "cv_R2": np.nan,
             **metrics(y_val, pred_val_bl), "train_R2": metrics(train[TARGET].values, pred_train_bl)["R2"]})
rows.append({"model": f"Ridge(alpha={ridge.alpha_:.2f})", "cv_MAE": np.nan, "cv_R2": np.nan,
             **metrics(y_val, pred_val_ridge), "train_R2": metrics(y_train, pred_train_ridge)["R2"]})
rows.append({"model": "RF (극단단순, depth2/n20/leaf10)", "cv_MAE": round(mae_simple_cv.mean(), 3),
             "cv_R2": round(r2_simple_cv.mean(), 3),
             **metrics(y_val, pred_val_simple), "train_R2": metrics(y_train, pred_train_simple)["R2"]})
rows.append({"model": f"RF (재탐색: depth{int(rf_pick['max_depth'])}/n{int(rf_pick['n_estimators'])}/leaf{int(rf_pick['min_samples_leaf'])})",
             "cv_MAE": round(rf_pick["cv_MAE_mean"], 3), "cv_R2": round(rf_pick["cv_R2_mean"], 3),
             **metrics(y_val, pred_val_rf), "train_R2": metrics(y_train, pred_train_rf)["R2"]})
rows.append({"model": f"GB (재탐색: depth{int(gb_pick['max_depth'])}/n{int(gb_pick['n_estimators'])}/leaf{int(gb_pick['min_samples_leaf'])}/lr{gb_pick['learning_rate']})",
             "cv_MAE": round(gb_pick["cv_MAE_mean"], 3), "cv_R2": round(gb_pick["cv_R2_mean"], 3),
             **metrics(y_val, pred_val_gb), "train_R2": metrics(y_train, pred_train_gb)["R2"]})

result_table = pd.DataFrame(rows)
print(result_table.to_string(index=False))

print("\n" + "=" * 90)
print("5. CV 평균 방향 vs Val 방향 일치 여부")
print("=" * 90)
for r in rows:
    if pd.isna(r["cv_R2"]):
        continue
    direction = "일치" if (r["cv_R2"] > 0) == (r["R2"] > 0) else "불일치(주의)"
    print(f"{r['model']}: CV_R2={r['cv_R2']}, Val_R2={r['R2']} -> {direction}")
