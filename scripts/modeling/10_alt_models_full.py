#!/usr/bin/env python3
"""
Items 1-7 requested together:
  1. Quantile Regression (linear, pinball loss) -- median (q=0.5) as point
     estimate, q=0.1/0.9 as an interval, on the standard 8-feature set.
  2. Feature reduction -- backward elimination from the 8 features by
     smallest |standardized Ridge coefficient|, tracking Val R2 at each step.
  3. Log-transformed target -- y' = sign(y)*log1p(|y|) (signed log, since
     WAR can be negative), Ridge on y', back-transformed for evaluation.
  4. SVR (RBF kernel), narrow CV grid over C/gamma.
  5. k-NN regression, k in 5-15 via CV.
  6. Gaussian Process Regression (RBF + white noise), reports predictive std.
  7. Master comparison table: all models (old + new) on Val(2024) + repeated
     CV (3-fold x 5 seeds, matching the protocol from 05_tree_retune.py),
     with the established CV/Val-direction-agreement check.

NOTE: items 1-3 were requested in a message this script's author could not
find in the visible conversation/repo history -- definitions above are
explicit, stated assumptions, not a continuation of prior verified work.
"""
import itertools
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, QuantileRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RepeatedKFold, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = "/Users/ojaemin/Downloads/kbo-pitcher-scouting"
TARGET = "kbo_first_year_WAR"

df = pd.read_csv(f"{ROOT}/data/rosters/analysis_dataset_v1.csv")
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

X_train_raw = train_fit[FEATURE_COLS].values
y_train = train_fit[TARGET].values
X_val_raw = val_fit[FEATURE_COLS].values
y_val = val_fit[TARGET].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_val = scaler.transform(X_val_raw)

rkf = RepeatedKFold(n_splits=3, n_repeats=5, random_state=42)


def metrics(y_true, y_pred):
    return {"MAE": round(mean_absolute_error(y_true, y_pred), 3),
            "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 3),
            "R2": round(r2_score(y_true, y_pred), 3)}


def pred_range(y_pred):
    return f"{y_pred.min():.2f}~{y_pred.max():.2f}"


master_rows = []

# ---------------------------------------------------------------------
# reference: Baseline / Ridge(8var) -- recomputed for a consistent table
# ---------------------------------------------------------------------
def baseline_feature(d):
    return d["mlb_fip_last"].where(d["mlb_fip_last"].notna(), d["aaa_fip_last"])


train["baseline_fip"] = baseline_feature(train)
val_fit_bl = val_f.copy()
val_fit_bl["baseline_fip"] = baseline_feature(val_fit_bl)
train_bl_fit = train.dropna(subset=["baseline_fip", TARGET])
bl_model = LinearRegression().fit(train_bl_fit[["baseline_fip"]].values, train_bl_fit[TARGET].values)
pred_val_bl = np.where(val_fit_bl["baseline_fip"].notna(),
                        bl_model.predict(val_fit_bl[["baseline_fip"]].fillna(0).values),
                        train_bl_fit[TARGET].mean())
master_rows.append({"model": "Baseline(단일FIP)", "cv_R2": np.nan,
                     **metrics(val_fit_bl[TARGET].values, pred_val_bl),
                     "pred_range": pred_range(pred_val_bl)})

ridge_cv5 = KFold(n_splits=5, shuffle=True, random_state=42)
alphas = np.logspace(-2, 3, 60)
ridge = RidgeCV(alphas=alphas, cv=ridge_cv5, scoring="neg_mean_absolute_error")
ridge.fit(X_train, y_train)
pred_val_ridge = ridge.predict(X_val)
cv_r2_ridge = cross_val_score(Ridge(alpha=ridge.alpha_), X_train, y_train, cv=rkf, scoring="r2")
master_rows.append({"model": f"Ridge(8var, alpha={ridge.alpha_:.2f})", "cv_R2": round(cv_r2_ridge.mean(), 3),
                     **metrics(y_val, pred_val_ridge), "pred_range": pred_range(pred_val_ridge)})

# RF/GB reference (extreme-simple + retuned), same protocol as 05_tree_retune.py
rf_simple = RandomForestRegressor(max_depth=2, n_estimators=20, min_samples_leaf=10, random_state=42)
rf_simple.fit(X_train_raw, y_train)
pred_val_rf_simple = rf_simple.predict(X_val_raw)
cv_r2_rf_simple = cross_val_score(rf_simple, X_train_raw, y_train, cv=rkf, scoring="r2")
master_rows.append({"model": "RF(극단단순 depth2/n20/leaf10)", "cv_R2": round(cv_r2_rf_simple.mean(), 3),
                     **metrics(y_val, pred_val_rf_simple), "pred_range": pred_range(pred_val_rf_simple)})

grid = {"max_depth": [2, 3], "n_estimators": [20, 50, 100], "min_samples_leaf": [10, 15, 20]}
combos = list(itertools.product(grid["max_depth"], grid["n_estimators"], grid["min_samples_leaf"]))
rf_rows, gb_rows = [], []
for depth, n_est, leaf in combos:
    m = RandomForestRegressor(max_depth=depth, n_estimators=n_est, min_samples_leaf=leaf, random_state=42)
    r2s = cross_val_score(m, X_train_raw, y_train, cv=rkf, scoring="r2")
    maes = -cross_val_score(m, X_train_raw, y_train, cv=rkf, scoring="neg_mean_absolute_error")
    rf_rows.append({"depth": depth, "n_estimators": n_est, "leaf": leaf, "cv_R2": r2s.mean(), "cv_MAE": maes.mean()})
    for lr in [0.05, 0.1]:
        m2 = GradientBoostingRegressor(max_depth=depth, n_estimators=n_est, min_samples_leaf=leaf,
                                        learning_rate=lr, random_state=42)
        r2s2 = cross_val_score(m2, X_train_raw, y_train, cv=rkf, scoring="r2")
        maes2 = -cross_val_score(m2, X_train_raw, y_train, cv=rkf, scoring="neg_mean_absolute_error")
        gb_rows.append({"depth": depth, "n_estimators": n_est, "leaf": leaf, "lr": lr,
                         "cv_R2": r2s2.mean(), "cv_MAE": maes2.mean()})
rf_best = min(rf_rows, key=lambda r: r["cv_MAE"])
gb_best = min(gb_rows, key=lambda r: r["cv_MAE"])
rf_final = RandomForestRegressor(max_depth=rf_best["depth"], n_estimators=rf_best["n_estimators"],
                                  min_samples_leaf=rf_best["leaf"], random_state=42).fit(X_train_raw, y_train)
gb_final = GradientBoostingRegressor(max_depth=gb_best["depth"], n_estimators=gb_best["n_estimators"],
                                      min_samples_leaf=gb_best["leaf"], learning_rate=gb_best["lr"],
                                      random_state=42).fit(X_train_raw, y_train)
pred_val_rf = rf_final.predict(X_val_raw)
pred_val_gb = gb_final.predict(X_val_raw)
master_rows.append({"model": f"RF(재탐색 d{rf_best['depth']}/n{rf_best['n_estimators']}/leaf{rf_best['leaf']})",
                     "cv_R2": round(rf_best["cv_R2"], 3), **metrics(y_val, pred_val_rf), "pred_range": pred_range(pred_val_rf)})
master_rows.append({"model": f"GB(재탐색 d{gb_best['depth']}/n{gb_best['n_estimators']}/leaf{gb_best['leaf']}/lr{gb_best['lr']})",
                     "cv_R2": round(gb_best["cv_R2"], 3), **metrics(y_val, pred_val_gb), "pred_range": pred_range(pred_val_gb)})

# ---------------------------------------------------------------------
# 1. Quantile Regression (linear, pinball loss)
# ---------------------------------------------------------------------
print("=" * 90)
print("1. Quantile Regression (선형, pinball loss)")
print("=" * 90)

qr_alphas = [0.001, 0.01, 0.1, 1.0, 10.0]
qr_cv_rows = []
for a in qr_alphas:
    m = QuantileRegressor(quantile=0.5, alpha=a, solver="highs")
    maes = -cross_val_score(m, X_train, y_train, cv=rkf, scoring="neg_mean_absolute_error")
    r2s = cross_val_score(m, X_train, y_train, cv=rkf, scoring="r2")
    qr_cv_rows.append({"alpha": a, "cv_MAE": maes.mean(), "cv_R2": r2s.mean()})
qr_table = pd.DataFrame(qr_cv_rows)
print(qr_table.round(4).to_string(index=False))
qr_best_alpha = qr_table.loc[qr_table["cv_MAE"].idxmin(), "alpha"]
print(f"-> 선택된 alpha={qr_best_alpha}")

qr_median = QuantileRegressor(quantile=0.5, alpha=qr_best_alpha, solver="highs").fit(X_train, y_train)
qr_lo = QuantileRegressor(quantile=0.1, alpha=qr_best_alpha, solver="highs").fit(X_train, y_train)
qr_hi = QuantileRegressor(quantile=0.9, alpha=qr_best_alpha, solver="highs").fit(X_train, y_train)
pred_val_qr = qr_median.predict(X_val)
pred_val_qr_lo = qr_lo.predict(X_val)
pred_val_qr_hi = qr_hi.predict(X_val)
cv_r2_qr = cross_val_score(QuantileRegressor(quantile=0.5, alpha=qr_best_alpha, solver="highs"),
                            X_train, y_train, cv=rkf, scoring="r2")
print(f"\nVal: {metrics(y_val, pred_val_qr)}, 예측범위 {pred_range(pred_val_qr)}")
print(f"10~90% 구간 평균 폭: {(pred_val_qr_hi - pred_val_qr_lo).mean():.2f} WAR")
master_rows.append({"model": f"Quantile Reg(median, alpha={qr_best_alpha})", "cv_R2": round(cv_r2_qr.mean(), 3),
                     **metrics(y_val, pred_val_qr), "pred_range": pred_range(pred_val_qr)})

# ---------------------------------------------------------------------
# 2. Feature reduction -- backward elimination on |standardized coef|
# ---------------------------------------------------------------------
print("\n" + "=" * 90)
print("2. 변수축소 (backward elimination, |표준화계수| 최소부터 제거)")
print("=" * 90)

remaining = list(FEATURE_COLS)
elim_rows = []
while len(remaining) >= 2:
    idx = [FEATURE_COLS.index(c) for c in remaining]
    Xt = X_train[:, idx]
    Xv = X_val[:, idx]
    r = RidgeCV(alphas=alphas, cv=ridge_cv5, scoring="neg_mean_absolute_error").fit(Xt, y_train)
    pv = r.predict(Xv)
    cv_r2 = cross_val_score(Ridge(alpha=r.alpha_), Xt, y_train, cv=rkf, scoring="r2")
    elim_rows.append({"n_features": len(remaining), "features": ",".join(remaining),
                       "alpha": round(r.alpha_, 2), "cv_R2": round(cv_r2.mean(), 3),
                       **metrics(y_val, pv)})
    coefs = dict(zip(remaining, np.abs(r.coef_)))
    drop = min(coefs, key=coefs.get)
    remaining.remove(drop)

elim_table = pd.DataFrame(elim_rows)
print(elim_table[["n_features", "alpha", "cv_R2", "R2", "MAE"]].to_string(index=False))
best_row = elim_table.loc[elim_table["cv_R2"].idxmax()]
print(f"\ncv_R2 기준 최선: n_features={best_row['n_features']}, features={best_row['features']}")
master_rows.append({"model": f"Ridge(변수축소, n={int(best_row['n_features'])})", "cv_R2": best_row["cv_R2"],
                     "MAE": best_row["MAE"], "RMSE": best_row["RMSE"], "R2": best_row["R2"],
                     "pred_range": "-"})

# ---------------------------------------------------------------------
# 3. Log-transformed target: y' = sign(y) * log1p(|y|)
# ---------------------------------------------------------------------
print("\n" + "=" * 90)
print("3. 로그변환 타깃 (y' = sign(y)*log1p(|y|), Ridge 학습 후 역변환)")
print("=" * 90)


def signed_log1p(y):
    return np.sign(y) * np.log1p(np.abs(y))


def inv_signed_log1p(y):
    return np.sign(y) * (np.expm1(np.abs(y)))


y_train_log = signed_log1p(y_train)
ridge_log = RidgeCV(alphas=alphas, cv=ridge_cv5, scoring="neg_mean_absolute_error").fit(X_train, y_train_log)
pred_val_log_raw = ridge_log.predict(X_val)
pred_val_log = inv_signed_log1p(pred_val_log_raw)


def cv_r2_on_log_target(model_cls, alpha, X, y_orig):
    scores = []
    for tr_idx, te_idx in rkf.split(X):
        m = model_cls(alpha=alpha).fit(X[tr_idx], signed_log1p(y_orig[tr_idx]))
        pred = inv_signed_log1p(m.predict(X[te_idx]))
        scores.append(r2_score(y_orig[te_idx], pred))
    return np.array(scores)


cv_r2_log = cv_r2_on_log_target(Ridge, ridge_log.alpha_, X_train, y_train)
print(f"alpha={ridge_log.alpha_:.2f}")
print(f"Val (원 단위로 역변환 후): {metrics(y_val, pred_val_log)}, 예측범위 {pred_range(pred_val_log)}")
print(f"CV R2 (역변환 기준): mean={cv_r2_log.mean():.3f}, std={cv_r2_log.std():.3f}")
master_rows.append({"model": f"Ridge(로그변환 타깃, alpha={ridge_log.alpha_:.2f})", "cv_R2": round(cv_r2_log.mean(), 3),
                     **metrics(y_val, pred_val_log), "pred_range": pred_range(pred_val_log)})

# ---------------------------------------------------------------------
# 4. SVR (RBF)
# ---------------------------------------------------------------------
print("\n" + "=" * 90)
print("4. SVR (RBF 커널)")
print("=" * 90)

svr_grid = list(itertools.product([0.5, 1, 2, 5, 10], ["scale", 0.01, 0.05, 0.1]))
svr_rows = []
for C, gamma in svr_grid:
    m = SVR(kernel="rbf", C=C, gamma=gamma)
    maes = -cross_val_score(m, X_train, y_train, cv=rkf, scoring="neg_mean_absolute_error")
    r2s = cross_val_score(m, X_train, y_train, cv=rkf, scoring="r2")
    svr_rows.append({"C": C, "gamma": gamma, "cv_MAE": maes.mean(), "cv_R2": r2s.mean()})
svr_table = pd.DataFrame(svr_rows).sort_values("cv_MAE")
print(svr_table.head(8).round(4).to_string(index=False))
svr_best = svr_table.iloc[0]
svr_final = SVR(kernel="rbf", C=svr_best["C"], gamma=svr_best["gamma"]).fit(X_train, y_train)
pred_val_svr = svr_final.predict(X_val)
print(f"\n선택: C={svr_best['C']}, gamma={svr_best['gamma']}")
print(f"Val: {metrics(y_val, pred_val_svr)}, 예측범위 {pred_range(pred_val_svr)}")
master_rows.append({"model": f"SVR(RBF, C={svr_best['C']}, gamma={svr_best['gamma']})",
                     "cv_R2": round(svr_best["cv_R2"], 3), **metrics(y_val, pred_val_svr),
                     "pred_range": pred_range(pred_val_svr)})

# ---------------------------------------------------------------------
# 5. k-NN regression
# ---------------------------------------------------------------------
print("\n" + "=" * 90)
print("5. k-NN 회귀 (k=5~15)")
print("=" * 90)

knn_rows = []
for k in range(5, 16):
    m = KNeighborsRegressor(n_neighbors=k)
    maes = -cross_val_score(m, X_train, y_train, cv=rkf, scoring="neg_mean_absolute_error")
    r2s = cross_val_score(m, X_train, y_train, cv=rkf, scoring="r2")
    knn_rows.append({"k": k, "cv_MAE": maes.mean(), "cv_R2": r2s.mean()})
knn_table = pd.DataFrame(knn_rows)
print(knn_table.round(4).to_string(index=False))
knn_best_k = int(knn_table.loc[knn_table["cv_MAE"].idxmin(), "k"])
knn_final = KNeighborsRegressor(n_neighbors=knn_best_k).fit(X_train, y_train)
pred_val_knn = knn_final.predict(X_val)
cv_r2_knn = knn_table.loc[knn_table["k"] == knn_best_k, "cv_R2"].iloc[0]
print(f"\n선택: k={knn_best_k}")
print(f"Val: {metrics(y_val, pred_val_knn)}, 예측범위 {pred_range(pred_val_knn)}")
master_rows.append({"model": f"kNN(k={knn_best_k})", "cv_R2": round(cv_r2_knn, 3),
                     **metrics(y_val, pred_val_knn), "pred_range": pred_range(pred_val_knn)})

# ---------------------------------------------------------------------
# 6. Gaussian Process Regression
# ---------------------------------------------------------------------
print("\n" + "=" * 90)
print("6. Gaussian Process Regression (RBF + WhiteKernel)")
print("=" * 90)

kernel = ConstantKernel(1.0) * RBF(length_scale=np.ones(X_train.shape[1])) + WhiteKernel(noise_level=1.0)
gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3, random_state=42)
gpr.fit(X_train, y_train)
pred_val_gpr, pred_val_gpr_std = gpr.predict(X_val, return_std=True)
cv_r2_gpr = cross_val_score(GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=42),
                             X_train, y_train, cv=rkf, scoring="r2")
print(f"학습된 커널: {gpr.kernel_}")
print(f"Val: {metrics(y_val, pred_val_gpr)}, 예측범위 {pred_range(pred_val_gpr)}")
print(f"예측 표준편차(신뢰구간 대용): 평균={pred_val_gpr_std.mean():.2f}, "
      f"범위={pred_val_gpr_std.min():.2f}~{pred_val_gpr_std.max():.2f}")
print(f"CV R2: mean={cv_r2_gpr.mean():.3f}, std={cv_r2_gpr.std():.3f}")
master_rows.append({"model": "Gaussian Process(RBF+White)", "cv_R2": round(cv_r2_gpr.mean(), 3),
                     **metrics(y_val, pred_val_gpr), "pred_range": pred_range(pred_val_gpr)})

# ---------------------------------------------------------------------
# 7. Master comparison table + CV/Val direction check
# ---------------------------------------------------------------------
print("\n" + "=" * 90)
print("7. 종합 비교표 (Val 2024 기준 + 반복 CV, n_train=130)")
print("=" * 90)
master = pd.DataFrame(master_rows)
print(master.to_string(index=False))

print("\n--- CV 방향 vs Val 방향 일치 여부 (둘 다 있는 모델만) ---")
for r in master_rows:
    if pd.isna(r.get("cv_R2", np.nan)) or "R2" not in r:
        continue
    direction = "일치" if (r["cv_R2"] > 0) == (r["R2"] > 0) else "불일치(주의)"
    print(f"{r['model']}: CV_R2={r['cv_R2']}, Val_R2={r['R2']} -> {direction}")

master.to_csv(f"{ROOT}/reports/modeling/alt_models_comparison.csv", index=False)
print(f"\n저장: {ROOT}/reports/modeling/alt_models_comparison.csv")
