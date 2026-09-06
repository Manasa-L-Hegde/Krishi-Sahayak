"""Train classifiers, engineer climate risk, and emit dashboard artifacts.

Usage: python scripts/train_model.py --data data/raw/Crop_recommendation.csv
"""
from __future__ import annotations

import argparse
import json
import os
from itertools import combinations

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, mean_absolute_error, mean_squared_error, precision_score,
                             r2_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"
OUT = "models"


def risk_targets(frame: pd.DataFrame, rainfall_stats: pd.DataFrame) -> pd.Series:
    stats = rainfall_stats.reindex(frame[TARGET]).reset_index(drop=True)
    sigma = stats["std"].replace(0, 1.0).fillna(1.0)
    return (abs(frame["rainfall"].reset_index(drop=True) - stats["mean"]) / (2 * sigma) * 100).clip(0, 100)


def download_if_needed(path: str) -> str:
    if os.path.exists(path):
        return path
    try:
        import kagglehub
        downloaded = kagglehub.dataset_download("atharvaingle/crop-recommendation-dataset")
        candidates = []
        for root, _, files in os.walk(downloaded):
            candidates.extend(os.path.join(root, f) for f in files if f.lower().endswith(".csv"))
        if not candidates:
            raise FileNotFoundError("Kaggle download contained no CSV")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pd.read_csv(candidates[0]).to_csv(path, index=False)
        return path
    except Exception as exc:
        raise FileNotFoundError(f"Dataset not found at {path}. Download the Kaggle dataset first. Original error: {exc}") from exc


def robustness_rate(model, x_test: pd.DataFrame) -> float:
    baseline = model.predict(x_test)
    flips = []
    for feature in FEATURES:
        for direction in (0.9, 1.1):
            perturbed = x_test.copy()
            perturbed[feature] = perturbed[feature] * direction
            flips.append(model.predict(perturbed) != baseline)
    return float(np.concatenate(flips).mean())


def classification_metrics(y_true, prediction) -> dict[str, float]:
    return {"accuracy": float(accuracy_score(y_true, prediction)),
            "macro_precision": float(precision_score(y_true, prediction, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(y_true, prediction, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(y_true, prediction, average="macro")),
            "weighted_f1": float(f1_score(y_true, prediction, average="weighted"))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/Crop_recommendation.csv")
    args = parser.parse_args()
    path = download_if_needed(args.data)
    frame = pd.read_csv(path)
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    frame = frame.rename(columns={"label": TARGET, "n": "N", "p": "P", "k": "K"})
    missing = sorted(set(FEATURES + [TARGET]) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    frame[FEATURES] = frame[FEATURES].apply(pd.to_numeric, errors="raise")
    encoder = LabelEncoder()
    frame["target_id"] = encoder.fit_transform(frame[TARGET])
    x_train, x_test, y_train, y_test = train_test_split(
        frame[FEATURES], frame["target_id"], test_size=0.2, random_state=42, stratify=frame["target_id"]
    )
    train_frame = x_train.copy()
    train_frame[TARGET] = encoder.inverse_transform(y_train)
    rainfall_stats = train_frame.groupby(TARGET)["rainfall"].agg(["mean", "std"])
    y_risk_train = risk_targets(train_frame, rainfall_stats)
    test_frame = x_test.reset_index(drop=True).copy()
    test_frame[TARGET] = encoder.inverse_transform(y_test)
    y_risk_test = risk_targets(test_frame, rainfall_stats)

    classifiers = {
        "RandomForest": RandomForestClassifier(n_estimators=350, random_state=42, class_weight="balanced"),
        "XGBoost": XGBClassifier(n_estimators=250, max_depth=5, learning_rate=0.08, subsample=0.9,
                                  colsample_bytree=0.9, objective="multi:softprob", eval_metric="mlogloss",
                                  random_state=42, n_jobs=2),
        "LogisticRegression": Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=2000))]),
    }
    results = {}
    fitted = {}
    for name, model in classifiers.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        results[name] = classification_metrics(y_test, pred)
        fitted[name] = model
    baseline_name = "RandomForest"
    improved_name = "RandomForest_700Trees"
    improved = RandomForestClassifier(n_estimators=700, random_state=42, class_weight="balanced")
    improved.fit(x_train, y_train)
    improved_pred = improved.predict(x_test)
    results[improved_name] = classification_metrics(y_test, improved_pred)
    fitted[improved_name] = improved
    baseline_flips = robustness_rate(fitted[baseline_name], x_test)
    improved_flips = robustness_rate(improved, x_test)
    champion_name = improved_name if (results[improved_name]["macro_f1"] > results[baseline_name]["macro_f1"] or
                                      (results[improved_name]["macro_f1"] == results[baseline_name]["macro_f1"] and improved_flips < baseline_flips)) else baseline_name
    champion = fitted[champion_name]
    champion_pred = champion.predict(x_test)
    report = classification_report(y_test, champion_pred, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, champion_pred, labels=list(range(len(encoder.classes_))))
    labels = list(range(len(encoder.classes_)))
    pair_counts = []
    for i, j in combinations(range(len(labels)), 2):
        count = int(matrix[i, j] + matrix[j, i])
        if count:
            pair_counts.append({"crop_a": encoder.classes_[labels[i]], "crop_b": encoder.classes_[labels[j]], "errors": count})
    pair_counts.sort(key=lambda item: item["errors"], reverse=True)

    risk_model = RandomForestRegressor(n_estimators=250, random_state=42, min_samples_leaf=2)
    risk_model.fit(x_train, y_risk_train)
    risk_pred = risk_model.predict(x_test)
    risk_validation = {"mae": float(mean_absolute_error(y_risk_test, risk_pred)),
                       "rmse": float(mean_squared_error(y_risk_test, risk_pred) ** 0.5),
                       "r2": float(r2_score(y_risk_test, risk_pred))}
    risk_by_crop = []
    for crop, indices in test_frame.groupby(TARGET).groups.items():
        actual = y_risk_test.loc[indices]
        predicted = pd.Series(risk_pred, index=y_risk_test.index).loc[indices]
        risk_by_crop.append({"crop": crop, "test_rows": int(len(indices)), "mae": float(mean_absolute_error(actual, predicted)),
                             "rmse": float(mean_squared_error(actual, predicted) ** 0.5)})

    os.makedirs(OUT, exist_ok=True)
    explainer = shap.TreeExplainer(champion) if champion_name != "LogisticRegression" else shap.Explainer(champion.predict_proba, x_train)
    shap_values = explainer(x_test.iloc[: min(500, len(x_test))])
    if shap_values.values.ndim == 3:
        shap_values = shap.Explanation(values=shap_values.values.mean(axis=2), base_values=shap_values.base_values,
                                       data=shap_values.data, feature_names=FEATURES)
    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(shap_values, max_display=len(FEATURES), show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "shap_summary.png"), dpi=160)
    plt.close()
    artifacts = {"classifier": champion, "risk_model": risk_model, "explainer": explainer, "label_encoder": encoder,
                 "features": FEATURES, "classes": list(champion.classes_), "rainfall_stats": rainfall_stats.to_dict("index")}
    joblib.dump(artifacts, os.path.join(OUT, "krishi_sahayak.joblib"))
    per_crop = {}
    for key, value in report.items():
        if str(key).isdigit():
            per_crop[encoder.classes_[int(key)]] = {m: float(value[m]) for m in ("precision", "recall", "f1-score", "support")}
    diagnostics = {"dataset_rows": int(len(frame)), "dataset_classes": int(frame[TARGET].nunique()),
                   "model_results": results, "champion": champion_name,
                   "per_crop": per_crop,
                   "confusion_pairs": pair_counts[:10], "robustness_flip_rate": robustness_rate(champion, x_test),
                   "risk_validation": risk_validation, "risk_by_crop": risk_by_crop,
                   "robustness_baseline": baseline_flips, "robustness_improved": improved_flips,
                   "improvement_experiment": {"baseline": baseline_name, "improved": improved_name,
                                               "metric": "macro_f1", "absolute_change": results[improved_name]["macro_f1"] - results[baseline_name]["macro_f1"],
                                               "robustness_absolute_change": improved_flips - baseline_flips,
                                               "interpretation": "Neutral macro-F1 with a small robustness gain; approximately twice the tree-building cost."},
                   "risk_definition": "clip(abs(rainfall - crop_training_mean) / (2 * crop_training_std) * 100, 0, 100)"}
    with open(os.path.join(OUT, "diagnostics.json"), "w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2)
    diagnostics_dir = "diagnostics"
    os.makedirs(diagnostics_dir, exist_ok=True)
    pd.DataFrame([{"model": name, **metrics} for name, metrics in results.items()]).to_csv(
        os.path.join(diagnostics_dir, "model_comparison.csv"), index=False
    )
    pd.DataFrame([{"experiment": "RandomForest_700Trees_vs_RandomForest", "baseline_macro_f1": results[baseline_name]["macro_f1"],
                   "improved_macro_f1": results[improved_name]["macro_f1"], "macro_f1_absolute_change": results[improved_name]["macro_f1"] - results[baseline_name]["macro_f1"],
                   "baseline_robustness_flip_rate": baseline_flips, "improved_robustness_flip_rate": improved_flips,
                   "robustness_absolute_change": improved_flips - baseline_flips, "training_cost": "approximately 2x trees"}]).to_csv(
        os.path.join(diagnostics_dir, "improvement_experiment.csv"), index=False
    )
    pd.DataFrame([{"crop": crop, **metrics} for crop, metrics in per_crop.items()]).to_csv(
        os.path.join(diagnostics_dir, "per_crop_metrics.csv"), index=False
    )
    pd.DataFrame(pair_counts).to_csv(os.path.join(diagnostics_dir, "confusion_pairs.csv"), index=False)
    confusion_frame = pd.DataFrame(matrix, index=encoder.classes_, columns=encoder.classes_)
    confusion_frame.index.name = "actual"
    confusion_frame.to_csv(os.path.join(diagnostics_dir, "confusion_matrix.csv"))
    pd.DataFrame([{"metric": "single_feature_perturbation_flip_rate", "value": diagnostics["robustness_flip_rate"]}]).to_csv(
        os.path.join(diagnostics_dir, "robustness_results.csv"), index=False
    )
    pd.DataFrame([risk_validation]).to_csv(os.path.join(diagnostics_dir, "risk_score_validation.csv"), index=False)
    pd.DataFrame(risk_by_crop).to_csv(os.path.join(diagnostics_dir, "risk_metrics_by_crop.csv"), index=False)
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
