#!/usr/bin/env python3
"""
Compute predicted_WAR and Translation Gap (actual - predicted) for all 165
players with a valid target, using the final saved Ridge model as-is.

Caveat carried into the output/report: predictions for the 130 train-period
players come from a model that was fit on them (some in-sample optimism),
while val(2024)/test(2025) predictions are genuinely out-of-sample. Gap
values are still meaningful as "how much did actual performance exceed a
reasonable expectation from overseas record" for every player, but this
asymmetry is worth keeping in mind when comparing a train-period gap to a
holdout-period gap directly.
"""
import pickle
import pandas as pd

ROOT = "/Users/ojaemin/Downloads/kbo-pitcher-scouting"

with open(f"{ROOT}/reports/modeling/ridge_final.pkl", "rb") as f:
    artifact = pickle.load(f)

model, scaler = artifact["model"], artifact["scaler"]
FEATURE_COLS = artifact["feature_cols"]
LEVEL_FEATURES = artifact["level_features"]
OTHER_FEATURES = artifact["other_features"]
train_means, train_other_means = artifact["train_means"], artifact["train_other_means"]
TARGET = artifact["target"]


def build_features(d):
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


df = pd.read_csv(f"{ROOT}/data/rosters/analysis_dataset_v1.csv")
modeling_pop = df[df["kbo_no_appearance"] != 1].copy()

pop_f = build_features(modeling_pop)
X = scaler.transform(pop_f[FEATURE_COLS].values)
pop_f["predicted_WAR"] = model.predict(X)
pop_f["translation_gap"] = pop_f[TARGET] - pop_f["predicted_WAR"]
pop_f["split"] = pop_f["연도"].apply(lambda y: "train" if y <= 2023 else ("val" if y == 2024 else "test"))

df["predicted_WAR"] = ""
df["translation_gap"] = ""
df.loc[pop_f.index, "predicted_WAR"] = pop_f["predicted_WAR"].round(3)
df.loc[pop_f.index, "translation_gap"] = pop_f["translation_gap"].round(3)
df.to_csv(f"{ROOT}/data/rosters/analysis_dataset_v1.csv", index=False)

print(f"predicted_WAR, translation_gap 저장 완료 (n={len(pop_f)}, kbo_no_appearance=1 2명은 결측)")

show_cols = ["선수명", "연도", "split", TARGET, "predicted_WAR", "translation_gap"]
top10 = pop_f.sort_values("translation_gap", ascending=False).head(10)
bottom10 = pop_f.sort_values("translation_gap", ascending=True).head(10)

print("\n" + "=" * 78)
print("Translation Gap 상위 10명 (예측보다 훨씬 잘함)")
print("=" * 78)
print(top10[show_cols].round(3).to_string(index=False))

print("\n" + "=" * 78)
print("Translation Gap 하위 10명 (예측보다 훨씬 못함)")
print("=" * 78)
print(bottom10[show_cols].round(3).to_string(index=False))

print(f"\ngap 전체 분포: mean={pop_f['translation_gap'].mean():.3f}, "
      f"std={pop_f['translation_gap'].std():.3f}, "
      f"min={pop_f['translation_gap'].min():.3f}, max={pop_f['translation_gap'].max():.3f}")
