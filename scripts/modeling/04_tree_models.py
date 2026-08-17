#!/usr/bin/env python3
"""
Modeling step 4: Random Forest / Gradient Boosting vs Baseline vs Ridge,
all evaluated on the identical n=18 2024 validation set. n=130 train is
small for tree ensembles, so hyperparameter grids are kept deliberately
narrow and train/val R2 gap is reported explicitly per model to catch
overfitting rather than declaring a winner on val alone.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = "/Users/ojaemin/Downloads/kbo-pitcher-scouting"
df = pd.read_csv(f"{ROOT}/data/rosters/analysis_dataset_v1.csv")
TARGET = "kbo_first_year_WAR"

modeling_pop = df[df["kbo_no_appearance"] != 1].copy()
train = modeling_pop[modeling_pop["연도"] <= 2023].copy()
val = modeling_pop[modeling_pop["연도"] == 2024].copy()

# 8-feature set (aaa_fip_last dropped per prior diagnostic)
LEVEL_FEATURES = {
    "mlb": ["mlb_fip_last", "mlb_fip_minus_career"],
    "aaa": ["aaa_hr9_last", "aaa_bb9_3yr"],
}
OTHER_FEATURES = ["age_at_kbo_entry", "n_pitch_types_recorded"]
ALL_RAW_FEATURES = LEVEL_FEATURES["mlb"] + LEVEL_FEATURES["aaa"] + OTHER_FEATURES


def build_features(d, train_means, train_other_means):
    d = d.copy()
    d["has_mlb_record"] = (d["mlb_career_ip"].fillna(0) > 0).astype(int)
    d["has_aaa_record"] = (d["aaa_career_ip"].fillna(0) > 0).astype(int)
    for level, cols in LEVEL_FEATURES.items():
        has_col = f"has_{level}_record"
        for col in cols:
            d[col] = d[col].where(d[has_col] == 1, train_means[col])
            d[col] = d[col].fillna(train_means[col])
    for col in OTHER_FEATURES:
        d[col] = d[col].fillna(train_other_means[col])
    return d


train["has_mlb_record_tmp"] = (train["mlb_career_ip"].fillna(0) > 0).astype(int)
train["has_aaa_record_tmp"] = (train["aaa_career_ip"].fillna(0) > 0).astype(int)
train_means = {}
for level, cols in LEVEL_FEATURES.items():
    has_col = f"has_{level}_record_tmp"
    for col in cols:
        train_means[col] = train.loc[train[has_col] == 1, col].mean()
train_other_means = {col: train[col].mean() for col in OTHER_FEATURES}

train_f = build_features(train, train_means, train_other_means)
val_f = build_features(val, train_means, train_other_means)

FEATURE_COLS = ALL_RAW_FEATURES + ["has_mlb_record", "has_aaa_record"]  # 8 features
train_fit = train_f.dropna(subset=[TARGET])
val_fit = val_f.dropna(subset=[TARGET])

X_train_raw = train_fit[FEATURE_COLS].values
y_train = train_fit[TARGET].values
X_val_raw = val_fit[FEATURE_COLS].values
y_val = val_fit[TARGET].values

cv = KFold(n_splits=5, shuffle=True, random_state=42)


def metrics(y_true, y_pred):
    return {"MAE": round(mean_absolute_error(y_true, y_pred), 3),
            "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 3),
            "R2": round(r2_score(y_true, y_pred), 3)}


results = []
train_results = {}

# ---- Baseline (fair, same as before) ----
def baseline_feature(d):
    return d["mlb_fip_last"].where(d["mlb_fip_last"].notna(), d["aaa_fip_last"])


train["baseline_fip"] = baseline_feature(train)
val_fit["baseline_fip"] = baseline_feature(val_fit)
train_bl_fit = train.dropna(subset=["baseline_fip", TARGET])
baseline_model = LinearRegression().fit(train_bl_fit[["baseline_fip"]].values, train_bl_fit[TARGET].values)
pred_val_bl = np.where(
    val_fit["baseline_fip"].notna(),
    baseline_model.predict(val_fit[["baseline_fip"]].fillna(0).values),
    train_bl_fit[TARGET].mean(),
)
pred_train_bl = np.where(
    train["baseline_fip"].notna(),
    baseline_model.predict(train[["baseline_fip"]].fillna(0).values),
    train_bl_fit[TARGET].mean(),
)
results.append({"model": "Baseline", **metrics(y_val, pred_val_bl)})
train_results["Baseline"] = metrics(train[TARGET].values, pred_train_bl)

# ---- Ridge (8 features, scaled) ----
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train_raw)
X_val_s = scaler.transform(X_val_raw)
alphas = np.logspace(-2, 3, 60)
ridge = RidgeCV(alphas=alphas, cv=cv, scoring="neg_mean_absolute_error")
ridge.fit(X_train_s, y_train)
pred_val_ridge = ridge.predict(X_val_s)
pred_train_ridge = ridge.predict(X_train_s)
results.append({"model": f"Ridge (alpha={ridge.alpha_:.2f})", **metrics(y_val, pred_val_ridge)})
train_results[f"Ridge (alpha={ridge.alpha_:.2f})"] = metrics(y_train, pred_train_ridge)

# ---- Random Forest ----
rf_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [2, 3, 4, 5],
    "min_samples_leaf": [3, 5, 10],
}
rf_search = GridSearchCV(RandomForestRegressor(random_state=42), rf_grid, cv=cv,
                          scoring="neg_mean_absolute_error", n_jobs=-1)
rf_search.fit(X_train_raw, y_train)
rf = rf_search.best_estimator_
pred_val_rf = rf.predict(X_val_raw)
pred_train_rf = rf.predict(X_train_raw)
results.append({"model": f"Random Forest {rf_search.best_params_}", **metrics(y_val, pred_val_rf)})
train_results[f"Random Forest"] = metrics(y_train, pred_train_rf)

# ---- Gradient Boosting ----
gb_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [2, 3],
    "learning_rate": [0.01, 0.05, 0.1],
    "min_samples_leaf": [5, 10],
}
gb_search = GridSearchCV(GradientBoostingRegressor(random_state=42), gb_grid, cv=cv,
                          scoring="neg_mean_absolute_error", n_jobs=-1)
gb_search.fit(X_train_raw, y_train)
gb = gb_search.best_estimator_
pred_val_gb = gb.predict(X_val_raw)
pred_train_gb = gb.predict(X_train_raw)
results.append({"model": f"Gradient Boosting {gb_search.best_params_}", **metrics(y_val, pred_val_gb)})
train_results[f"Gradient Boosting"] = metrics(y_train, pred_train_gb)

print("=" * 90)
print("2. Validation(2024, n=18) 성능 비교 -- Baseline / Ridge / RF / GB")
print("=" * 90)
print(pd.DataFrame(results).to_string(index=False))

print(f"\nRandom Forest 최적 파라미터: {rf_search.best_params_}")
print(f"Gradient Boosting 최적 파라미터: {gb_search.best_params_}")

# ---- 4. train/val gap ----
print("\n" + "=" * 90)
print("4. Train vs Validation R2 격차 (과적합 점검)")
print("=" * 90)
gap_rows = []
short_names = ["Baseline", f"Ridge (alpha={ridge.alpha_:.2f})", "Random Forest", "Gradient Boosting"]
val_r2_map = {r["model"].split(" {")[0].split(" (")[0]: r["R2"] for r in results}
for name in short_names:
    key = name.split(" (")[0]
    tr = train_results[name]
    val_r2 = [r["R2"] for r in results if r["model"].startswith(name.split(" {")[0])][0]
    gap_rows.append({"model": name, "train_R2": tr["R2"], "val_R2": val_r2, "gap": round(tr["R2"] - val_r2, 3)})
print(pd.DataFrame(gap_rows).to_string(index=False))

# ---- 3. feature importance ----
print("\n" + "=" * 90)
print("3. Feature importance 비교 (Ridge |coef| 표준화 vs RF vs GB)")
print("=" * 90)
ridge_coef = pd.Series(np.abs(ridge.coef_), index=FEATURE_COLS).sort_values(ascending=False)
rf_imp = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
gb_imp = pd.Series(gb.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)

imp_table = pd.DataFrame({
    "Ridge |coef| 순위": ridge_coef.rank(ascending=False).astype(int),
    "RF importance 순위": rf_imp.reindex(FEATURE_COLS).rank(ascending=False).astype(int),
    "GB importance 순위": gb_imp.reindex(FEATURE_COLS).rank(ascending=False).astype(int),
    "RF importance": rf_imp.reindex(FEATURE_COLS).round(3),
    "GB importance": gb_imp.reindex(FEATURE_COLS).round(3),
    "Ridge |coef|": ridge_coef.reindex(FEATURE_COLS).round(3),
})
imp_table = imp_table.sort_values("RF importance 순위")
print(imp_table.to_string())
