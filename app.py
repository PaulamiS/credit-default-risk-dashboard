"""Streamlit dashboard for the credit default (delinquency) risk model.

Run `python train.py` first, then: streamlit run app.py
"""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

OUT_DIR = Path(__file__).parent / "outputs"
SEGMENT_COLORS = {"Low": "#2E86AB", "Medium": "#F2A541", "High": "#D1495B"}
SEGMENT_ORDER = ["Low", "Medium", "High"]
EDU_LABELS = {1: "Graduate school", 2: "University", 3: "High school", 4: "Other"}

st.set_page_config(page_title="Credit Default Risk Dashboard", page_icon=None, layout="wide")


@st.cache_data
def load_outputs():
    metrics = json.loads((OUT_DIR / "metrics.json").read_text())
    preds = pd.read_csv(OUT_DIR / "test_predictions.csv")
    shap_vals = pd.read_csv(OUT_DIR / "shap_values.csv")
    importance = pd.read_csv(OUT_DIR / "shap_importance.csv")
    return metrics, preds, shap_vals, importance


required = ["metrics.json", "test_predictions.csv", "shap_values.csv", "shap_importance.csv"]
if not all((OUT_DIR / f).exists() for f in required):
    st.title("Credit Default Risk Dashboard")
    st.error("Model outputs not found. Run `python train.py` first, then reload this page.")
    st.stop()

metrics, preds, shap_vals, importance = load_outputs()
preds["education_label"] = preds["EDUCATION"].map(EDU_LABELS).fillna("Other")

# ---------------------------------------------------------------- header
st.title("Credit Default Risk Dashboard")
st.caption(
    "XGBoost model predicting default on the next payment, with SHAP explanations. "
    "All figures are from a held-out test set the model never saw during training."
)

# ---------------------------------------------------------------- sidebar filters
st.sidebar.header("Filters")
segments = st.sidebar.multiselect("Risk segment", SEGMENT_ORDER, default=SEGMENT_ORDER)
lim_min, lim_max = int(preds["LIMIT_BAL"].min()), int(preds["LIMIT_BAL"].max())
limit_range = st.sidebar.slider("Credit limit", lim_min, lim_max, (lim_min, lim_max))
age_min, age_max = int(preds["AGE"].min()), int(preds["AGE"].max())
age_range = st.sidebar.slider("Age", age_min, age_max, (age_min, age_max))
edu_options = sorted(preds["education_label"].unique())
edu_sel = st.sidebar.multiselect("Education", edu_options, default=edu_options)

view = preds[
    preds["segment"].isin(segments)
    & preds["LIMIT_BAL"].between(*limit_range)
    & preds["AGE"].between(*age_range)
    & preds["education_label"].isin(edu_sel)
]
if view.empty:
    st.warning("No customers match the current filters.")
    st.stop()

# ---------------------------------------------------------------- KPIs
k1, k2, k3, k4 = st.columns(4)
k1.metric("Customers in view", f"{len(view):,}")
k2.metric("Actual default rate", f"{view['actual'].mean():.1%}")
k3.metric("Average predicted risk", f"{view['prob'].mean():.1%}")
k4.metric("Flagged high risk", f"{(view['segment'] == 'High').mean():.1%}")

tab_overview, tab_drivers, tab_customer, tab_model = st.tabs(
    ["Risk overview", "Risk drivers", "Customer explorer", "Model and fairness"]
)

# ---------------------------------------------------------------- overview
with tab_overview:
    seg = (
        view.groupby("segment")
        .agg(customers=("actual", "size"), default_rate=("actual", "mean"), defaults=("actual", "sum"))
        .reindex(SEGMENT_ORDER)
        .dropna()
        .reset_index()
    )
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            seg, x="segment", y="customers", color="segment", color_discrete_map=SEGMENT_COLORS,
            category_orders={"segment": SEGMENT_ORDER}, title="Customers by risk segment", text_auto=",",
        )
        fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Customers")
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = px.bar(
            seg, x="segment", y="default_rate", color="segment", color_discrete_map=SEGMENT_COLORS,
            category_orders={"segment": SEGMENT_ORDER}, title="Actual default rate by segment",
        )
        fig.update_traces(texttemplate="%{y:.1%}", textposition="outside")
        fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Actual default rate", yaxis_tickformat=".0%")
        st.plotly_chart(fig, width="stretch")

    fig = px.histogram(
        view, x="prob", color="segment", nbins=50, color_discrete_map=SEGMENT_COLORS,
        category_orders={"segment": SEGMENT_ORDER}, title="Distribution of predicted default probability",
    )
    fig.update_layout(xaxis_title="Predicted probability of default", yaxis_title="Customers", xaxis_tickformat=".0%")
    st.plotly_chart(fig, width="stretch")

    # Insight sentences are computed from the data in view, not hard-coded.
    high = view[view["segment"] == "High"]
    if len(high) and view["actual"].sum() > 0:
        st.info(
            f"**Business insight:** the High-risk segment is {len(high) / len(view):.1%} of the customers in view "
            f"but contains {high['actual'].sum() / view['actual'].sum():.1%} of all actual defaults. "
            f"Its default rate is {high['actual'].mean():.1%} versus {view['actual'].mean():.1%} overall, "
            "so collections effort can be prioritised there first."
        )

# ---------------------------------------------------------------- drivers
with tab_drivers:
    top = importance.head(15).iloc[::-1]
    fig = px.bar(top, x="mean_abs_shap", y="feature", orientation="h", title="Top 15 risk drivers (mean |SHAP value|)")
    fig.update_layout(xaxis_title="Average impact on the model's log-odds of default", yaxis_title=None, height=520)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "A larger bar means the feature moves the prediction more on average. "
        "Repayment-delay features (PAY_*, max_delay, months_delayed) are expected to dominate in this dataset."
    )
    png = OUT_DIR / "shap_summary.png"
    if png.exists():
        with st.expander("SHAP summary plot (direction of each feature's effect)"):
            st.image(str(png))

# ---------------------------------------------------------------- customer explorer
with tab_customer:
    st.write("Pick a customer to see why the model scored them as it did.")
    ranked = view.sort_values("prob", ascending=False)
    choice = st.selectbox(
        "Customer (sorted from highest to lowest predicted risk)",
        ranked["customer_id"].tolist(),
        format_func=lambda cid: f"Customer {cid}  |  predicted risk {ranked.loc[ranked['customer_id'] == cid, 'prob'].iloc[0]:.1%}",
    )
    row = preds[preds["customer_id"] == choice].iloc[0]
    s_row = shap_vals[shap_vals["customer_id"] == choice].iloc[0].drop("customer_id")

    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted default risk", f"{row['prob']:.1%}")
    m2.metric("Risk segment", row["segment"])
    m3.metric("Actual outcome", "Defaulted" if row["actual"] == 1 else "Paid on time")

    contrib = pd.DataFrame(
        {"feature": s_row.index, "shap_value": s_row.values, "feature_value": [row[f] for f in s_row.index]}
    )
    contrib = contrib.reindex(contrib["shap_value"].abs().sort_values(ascending=False).index).head(10)
    contrib["direction"] = contrib["shap_value"].apply(lambda v: "Raises risk" if v > 0 else "Lowers risk")
    contrib["label"] = contrib.apply(lambda r: f"{r['feature']} = {r['feature_value']:.2f}", axis=1)
    fig = px.bar(
        contrib.iloc[::-1], x="shap_value", y="label", orientation="h", color="direction",
        color_discrete_map={"Raises risk": "#D1495B", "Lowers risk": "#2E86AB"},
        title="Top 10 factors behind this customer's score",
    )
    fig.update_layout(xaxis_title="Contribution to log-odds of default", yaxis_title=None, height=480)
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- model and fairness
with tab_model:
    st.subheader("Model performance (held-out test set)")
    a, b, c, d = st.columns(4)
    a.metric("AUC-ROC", f"{metrics['test_auc_roc']:.3f}")
    b.metric("F1 score", f"{metrics['test_f1']:.3f}")
    c.metric("Precision", f"{metrics['test_precision']:.3f}")
    d.metric("Recall", f"{metrics['test_recall']:.3f}")
    st.caption(
        f"5-fold CV AUC on the training data: {metrics['cv_auc_mean']:.3f} +/- {metrics['cv_auc_std']:.3f}. "
        f"Logistic regression baseline AUC: {metrics['baseline_logreg_test_auc']:.3f}. "
        f"Decision threshold {metrics['decision_threshold']:.2f} was chosen on a validation split, not the test set."
    )

    st.subheader("Fairness check")
    st.write(
        "SEX is **not** a model input. This audit checks whether the model's high-risk flags still differ across groups. "
        "'Favorable' means not flagged high risk. A disparate impact ratio below 0.80 is a common warning sign, not a legal verdict."
    )
    for key, title in [("sex", "By sex"), ("age_group", "By age group")]:
        audit = metrics["fairness"][key]
        st.markdown(f"**{title}** | demographic parity difference: {audit['demographic_parity_difference']:.3f} | "
                    f"disparate impact ratio: {audit['disparate_impact_ratio']:.3f}")
        table = pd.DataFrame(audit["groups"]).rename(
            columns={
                "group": "Group", "customers": "Customers", "flagged_high_risk_rate": "Flagged high risk",
                "favorable_rate": "Favorable rate", "actual_default_rate": "Actual default rate",
            }
        )
        for col in ["Flagged high risk", "Favorable rate", "Actual default rate"]:
            table[col] = table[col].map("{:.1%}".format)
        st.dataframe(table, hide_index=True, width="stretch")

    st.caption(
        "Limitations: the data is from one credit card issuer in Taiwan (2005), so results illustrate the method and "
        "should not be used for real lending decisions."
    )
