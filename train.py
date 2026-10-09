"""Train an XGBoost credit-default (delinquency) risk model and export everything
the Streamlit dashboard needs.

Usage:
    python train.py                       # expects data/UCI_Credit_Card.csv
    python train.py --data path/to/file   # .csv, .xls or .xlsx
    python train.py --out outputs         # where artifacts are written

Dataset: "Default of Credit Card Clients" (UCI Machine Learning Repository,
also on Kaggle as UCI_Credit_Card.csv). 30,000 customers, 24 raw columns,
target = default payment next month.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

SEED = 42
TARGET = "default"
PAY_COLS = [f"PAY_{i}" for i in range(1, 7)]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYAMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]
RAW_COLS = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"] + PAY_COLS + BILL_COLS + PAYAMT_COLS
# SEX is NOT used as a model input. It is kept only to audit the model for fairness.
EXCLUDED_FROM_MODEL = ["SEX"]


# ----------------------------------------------------------------------------
# Data loading and cleaning
# ----------------------------------------------------------------------------
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(
            f"Dataset not found at {path}.\n"
            "Download 'Default of Credit Card Clients' (UCI or Kaggle: UCI_Credit_Card.csv) "
            "and place it in the data/ folder, or pass --data <path>."
        )
    if path.suffix.lower() in {".xls", ".xlsx"}:
        df = pd.read_excel(path, header=1)  # the UCI Excel file has a title row first
    else:
        df = pd.read_csv(path)

    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(
        columns={
            "PAY_0": "PAY_1",
            "default.payment.next.month": TARGET,
            "default payment next month": TARGET,
        }
    )
    df = df.drop(columns=["ID"], errors="ignore")

    missing = [c for c in RAW_COLS + [TARGET] if c not in df.columns]
    if missing:
        raise SystemExit(f"Dataset is missing expected columns: {missing}")

    df = df[RAW_COLS + [TARGET]].dropna().reset_index(drop=True)
    # Undocumented category codes in the original data are folded into 'other'.
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    limit = out["LIMIT_BAL"].replace(0, np.nan)

    out["avg_bill"] = out[BILL_COLS].mean(axis=1)
    out["avg_payment"] = out[PAYAMT_COLS].mean(axis=1)
    out["utilization_latest"] = (out["BILL_AMT1"] / limit).clip(-1, 3)
    out["utilization_avg"] = (out["avg_bill"] / limit).clip(-1, 3)
    total_bill = out[BILL_COLS].clip(lower=0).sum(axis=1).replace(0, np.nan)
    out["payment_to_bill_ratio"] = (out[PAYAMT_COLS].sum(axis=1) / total_bill).clip(0, 5).fillna(1.0)
    out["max_delay"] = out[PAY_COLS].max(axis=1)
    out["months_delayed"] = (out[PAY_COLS] > 0).sum(axis=1)
    out["delay_trend"] = out["PAY_1"] - out["PAY_6"]
    out["bill_trend"] = ((out["BILL_AMT1"] - out["BILL_AMT6"]) / limit).clip(-3, 3)
    return out.fillna(0)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def best_f1_threshold(y_true, prob) -> float:
    grid = np.linspace(0.05, 0.80, 76)
    scores = [f1_score(y_true, prob >= t) for t in grid]
    return float(grid[int(np.argmax(scores))])


def assign_segment(prob: np.ndarray, threshold: float) -> np.ndarray:
    """High = at or above the decision threshold, Medium = between half the
    threshold and the threshold, Low = below."""
    return np.where(prob >= threshold, "High", np.where(prob >= threshold / 2, "Medium", "Low"))


def group_audit(frame: pd.DataFrame, group_col: str, threshold: float) -> dict:
    """Demographic-parity style audit at the decision threshold.
    'favorable' = NOT flagged high risk."""
    rows = []
    for name, g in frame.groupby(group_col, observed=True):
        rows.append(
            {
                "group": str(name),
                "customers": int(len(g)),
                "flagged_high_risk_rate": float((g["prob"] >= threshold).mean()),
                "favorable_rate": float((g["prob"] < threshold).mean()),
                "actual_default_rate": float(g["actual"].mean()),
            }
        )
    fav = [r["favorable_rate"] for r in rows]
    flagged = [r["flagged_high_risk_rate"] for r in rows]
    return {
        "groups": rows,
        "demographic_parity_difference": float(max(flagged) - min(flagged)),
        "disparate_impact_ratio": float(min(fav) / max(fav)) if max(fav) > 0 else None,
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/UCI_Credit_Card.csv")
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_data(Path(args.data))
    full = add_features(raw)
    feature_cols = [c for c in full.columns if c not in [TARGET] + EXCLUDED_FROM_MODEL]
    X, y, sex = full[feature_cols], full[TARGET].astype(int), full["SEX"]
    print(f"Loaded {len(full):,} customers, default rate {y.mean():.1%}, {len(feature_cols)} features")

    X_train, X_test, y_train, y_test, _, sex_test = train_test_split(
        X, y, sex, test_size=0.2, stratify=y, random_state=SEED
    )
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=SEED
    )

    params = dict(
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=2.0,
        eval_metric="auc",
        random_state=SEED,
        n_jobs=-1,
    )
    model = XGBClassifier(n_estimators=600, early_stopping_rounds=30, **params)
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    best_n = int(model.best_iteration) + 1
    print(f"XGBoost stopped at {best_n} trees")

    # Decision threshold is chosen on the validation split, never on the test set.
    threshold = best_f1_threshold(y_val, model.predict_proba(X_val)[:, 1])

    # 5-fold cross-validated AUC on the training data.
    cv_model = XGBClassifier(n_estimators=best_n, **params)
    cv_auc = cross_val_score(
        cv_model, X_train, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring="roc_auc"
    )

    # Baseline for comparison.
    baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    baseline.fit(X_train, y_train)
    baseline_auc = roc_auc_score(y_test, baseline.predict_proba(X_test)[:, 1])

    # Final evaluation on the untouched test set.
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= threshold).astype(int)
    metrics = {
        "n_customers": int(len(full)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "overall_default_rate": float(y.mean()),
        "n_trees": best_n,
        "decision_threshold": threshold,
        "segment_cutoffs": {"medium_from": threshold / 2, "high_from": threshold},
        "test_auc_roc": float(roc_auc_score(y_test, prob)),
        "test_f1": float(f1_score(y_test, pred)),
        "test_precision": float(precision_score(y_test, pred)),
        "test_recall": float(recall_score(y_test, pred)),
        "test_average_precision": float(average_precision_score(y_test, prob)),
        "cv_auc_mean": float(cv_auc.mean()),
        "cv_auc_std": float(cv_auc.std()),
        "baseline_logreg_test_auc": float(baseline_auc),
        "features": feature_cols,
        "excluded_from_model": EXCLUDED_FROM_MODEL,
    }

    # Scored test set.
    scored = X_test.copy().reset_index(drop=True)
    scored.insert(0, "customer_id", X_test.index.to_numpy())
    scored["SEX"] = sex_test.to_numpy()
    scored["actual"] = y_test.to_numpy()
    scored["prob"] = prob
    scored["segment"] = assign_segment(prob, threshold)

    seg_rows = []
    total_defaults = int(scored["actual"].sum())
    for seg in ["Low", "Medium", "High"]:
        g = scored[scored["segment"] == seg]
        seg_rows.append(
            {
                "segment": seg,
                "customers": int(len(g)),
                "share_of_customers": float(len(g) / len(scored)),
                "actual_default_rate": float(g["actual"].mean()) if len(g) else None,
                "share_of_all_defaults": float(g["actual"].sum() / total_defaults) if total_defaults else None,
            }
        )
    metrics["segments"] = seg_rows

    # Fairness audit (SEX is not a model input, so this checks for indirect effects).
    audit_frame = scored.copy()
    audit_frame["sex_label"] = audit_frame["SEX"].map({1: "Male", 2: "Female"}).fillna("Other")
    audit_frame["age_group"] = pd.cut(
        audit_frame["AGE"], bins=[0, 29, 39, 49, 200], labels=["<30", "30-39", "40-49", "50+"]
    )
    metrics["fairness"] = {
        "sex": group_audit(audit_frame, "sex_label", threshold),
        "age_group": group_audit(audit_frame, "age_group", threshold),
    }

    # SHAP explanations for every test customer.
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    metrics["shap_base_value"] = float(np.ravel(explainer.expected_value)[0])
    shap_df = pd.DataFrame(shap_values, columns=feature_cols)
    shap_df.insert(0, "customer_id", X_test.index.to_numpy())
    importance = (
        pd.DataFrame({"feature": feature_cols, "mean_abs_shap": np.abs(shap_values).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Save everything.
    model.save_model(out_dir / "xgb_model.json")
    scored.to_csv(out_dir / "test_predictions.csv", index=False)
    shap_df.to_csv(out_dir / "shap_values.csv", index=False)
    importance.to_csv(out_dir / "shap_importance.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Ready-to-paste results block for the README.
    lines = [
        "## Results (held-out test set)",
        "",
        f"- Customers: {metrics['n_customers']:,} (test set: {metrics['n_test']:,}), overall default rate {metrics['overall_default_rate']:.1%}",
        f"- XGBoost AUC-ROC: {metrics['test_auc_roc']:.3f} (5-fold CV on train: {metrics['cv_auc_mean']:.3f} +/- {metrics['cv_auc_std']:.3f})",
        f"- Logistic regression baseline AUC-ROC: {metrics['baseline_logreg_test_auc']:.3f}",
        f"- At threshold {threshold:.2f}: F1 {metrics['test_f1']:.3f}, precision {metrics['test_precision']:.3f}, recall {metrics['test_recall']:.3f}",
        "",
        "| Risk segment | Customers | Share of customers | Actual default rate | Share of all defaults |",
        "|---|---|---|---|---|",
    ]
    def pct(value):
        return "n/a" if value is None else f"{value:.1%}"

    for r in seg_rows:
        lines.append(
            f"| {r['segment']} | {r['customers']:,} | {pct(r['share_of_customers'])} | "
            f"{pct(r['actual_default_rate'])} | {pct(r['share_of_all_defaults'])} |"
        )
    lines += ["", "Top risk drivers (mean |SHAP|): " + ", ".join(importance["feature"].head(5)) + "."]
    (out_dir / "results.md").write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nArtifacts written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
