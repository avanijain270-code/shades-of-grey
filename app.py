"""Shades of Grey - Business Validation Dashboard (Streamlit).
Run:  streamlit run app.py
"""
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analysis import run_all
from src.generate_data import generate
from src.schema import ITEMS, SURVEY
from src.viability import Assumptions, scorecard, unit_economics

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Shades of Grey | Validation Dashboard", page_icon="🩶", layout="wide")
GREYS = ["#1f1f1f", "#7a7a7a", "#c4c4c4", "#9b2c2c"]
SEG_COLORS = {"High probability": "#1f1f1f", "Medium probability": "#7a7a7a", "Low probability": "#c9c9c9"}

REQUIRED = ["q01_age_group", "q05_income_aed", "q07_clothing_spend_aed", "q10_grey_pref", "q13_concept_appeal",
            "q21_purchase_intent", "q25_spend_6m_aed", *ITEMS.keys()]


@st.cache_data(show_spinner="Running 4 algorithms...")
def analyse(df: pd.DataFrame):
    return run_all(df)


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🩶 Shades of Grey")
st.sidebar.caption("Idea validation with 4 ML algorithms")
source = st.sidebar.radio("Data source", ["Synthetic (simulated)", "Upload real survey CSV"])
if source.startswith("Synthetic"):
    n = st.sidebar.slider("Respondents", 200, 2000, 600, 50)
    noise = st.sidebar.slider("Noise & outliers (0 = clean, 2 = very messy)", 0.0, 2.0, 1.0, 0.25)
    seed = st.sidebar.number_input("Random seed", 0, 9999, 42)
    raw = generate(n=n, seed=int(seed), noise=noise)
    is_synth = True
else:
    up = st.sidebar.file_uploader("CSV with the same columns as the survey", type="csv")
    template = generate(n=50, seed=1, noise=0).to_csv(index=False)
    st.sidebar.download_button("Download column template", template, "survey_template.csv", "text/csv")
    if up is None:
        st.info("Upload your real survey export (same column names as the template) to re-run everything on real data.")
        st.stop()
    raw = pd.read_csv(up)
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        st.error(f"Missing columns: {missing[:8]}{'...' if len(missing) > 8 else ''}")
        st.stop()
    is_synth = False

R = analyse(raw)

st.sidebar.markdown("---")
st.sidebar.subheader("Business assumptions (edit!)")
A = Assumptions(
    reachable_audience=st.sidebar.number_input("Reachable audience, year 1", 10_000, 5_000_000, 150_000, 10_000),
    intent_to_purchase=st.sidebar.slider("Intent → real purchase", 0.05, 0.60, 0.25, 0.01),
    spend_haircut=st.sidebar.slider("Stated → real spend", 0.3, 1.0, 0.70, 0.05),
    planned_tshirt_price=st.sidebar.number_input("Planned T-shirt price (AED)", 20.0, 300.0, 59.0, 1.0),
    gross_margin=st.sidebar.slider("Gross margin", 0.25, 0.70, 0.50, 0.01),
    fulfilment_pct=st.sidebar.slider("Fulfilment % revenue", 0.0, 0.20, 0.09, 0.01),
    payment_pct=0.025,
    returns_loss_pct=st.sidebar.slider("Returns loss % revenue", 0.0, 0.20, 0.05, 0.01),
    annual_retention=st.sidebar.slider("Annual repeat rate", 0.1, 0.8, 0.40, 0.05),
    cac=st.sidebar.number_input("CAC - broad campaign (AED)", 10.0, 500.0, 90.0, 5.0),
    fixed_costs_year=st.sidebar.number_input("Fixed costs / year (AED)", 100_000.0, 10_000_000.0, 1_200_000.0, 50_000.0),
)
UE = unit_economics(R, A)
SC, passed, verdict, tone = scorecard(R, A, UE)
d, clf, clu, asr, reg = R["data"], R["clf"], R["clu"], R["asr"], R["reg"]

st.title("Shades of Grey - Is the idea viable?")
if is_synth:
    st.warning("**Data source: SYNTHETIC.** Patterns were simulated from our own assumptions, so this run proves the "
               "pipeline works - it does **not** prove customers want the brand. Upload the real survey to get the real answer.")

tabs = st.tabs(["✅ Verdict", "🧹 Data & noise", "1 Classification", "2 Clustering", "3 Association rules",
                "4 Regression", "💰 Unit economics", "🎯 Strategy", "📝 Survey (30 Qs)"])

# ------------------------------------------------------------------ Verdict
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Verdict", verdict)
    c2.metric("Checks passed", f"{passed} / {len(SC)}")
    c3.metric("Interested customers", f"{UE['p_interested']:.0%}")
    c4.metric("LTV : CAC", f"{UE['broad']['ltv_cac']:.1f}x")
    color = {"green": "success", "orange": "warning", "red": "error"}[tone]
    getattr(st, color)(f"**{verdict}** - {passed} of {len(SC)} viability checks pass under the current assumptions"
                       + (" (on simulated data)." if is_synth else "."))
    st.dataframe(SC.assign(Pass=SC["Pass"].map({True: "✅", False: "❌"})), hide_index=True, width="stretch")
    fails = SC[~SC["Pass"]]
    if len(fails):
        st.markdown("**What must be fixed or proven before launch:** " + "; ".join(fails["Check"].str.replace(r"^\d\. ", "", regex=True)) + ".")
    st.caption("Viable = enough demand + customers accept the price + the numbers repay acquisition and fixed costs. "
               "Thresholds are rules of thumb, editable in `src/viability.py`.")

# ------------------------------------------------------------------ Data & noise
with tabs[1]:
    rep = R["report"]
    a, b, c, e = st.columns(4)
    a.metric("Raw rows", rep["rows_raw"]); b.metric("Duplicates removed", rep["duplicates_removed"])
    c.metric("Careless (straight-line) removed", rep["careless_removed"]); e.metric("Missing cells imputed", rep["missing_cells"])
    st.markdown("**Why outliers matter:** a single typo (spend 8,000 instead of 800) can drag a regression line. "
                "We cap only *extreme* values (> Q3 + 3×IQR) so real big spenders survive.")
    col = st.selectbox("Inspect column", list(rep["outliers_capped"].keys()), index=1)
    fig = go.Figure()
    fig.add_box(y=raw[col], name="Before cleaning", marker_color="#9b2c2c", boxpoints="outliers")
    fig.add_box(y=d[col], name="After cleaning", marker_color="#1f1f1f", boxpoints="outliers")
    fig.update_layout(height=380, margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")
    st.write("Outliers capped per column:", rep["outliers_capped"])
    st.dataframe(raw.head(30), width="stretch")
    st.download_button("⬇ Download raw dataset", raw.to_csv(index=False), "survey_raw.csv", "text/csv")

# ------------------------------------------------------------------ Classification
with tabs[2]:
    st.subheader("Classification - will a person be interested?")
    st.caption("Target: purchase intent 4-5 (Q21). Answers to outcome questions (intent, subscription, NPS, spend) are excluded so the model cannot cheat.")
    m = clf["metrics"]
    st.dataframe(m.style.format("{:.3f}").highlight_max(axis=0, color="#d8d8d8"), width="stretch")
    st.success(f"Best model: **{clf['best']}** (5-fold AUC {m.loc[clf['best'], 'CV AUC (5-fold)']:.2f}). AUC 0.5 = coin flip, 1.0 = perfect.")
    l, r_ = st.columns(2)
    with l:
        fig = go.Figure()
        for nm, (fpr, tpr) in clf["roc"].items():
            fig.add_scatter(x=fpr, y=tpr, name=nm, mode="lines")
        fig.add_scatter(x=[0, 1], y=[0, 1], name="Coin flip", line=dict(dash="dash", color="#bbb"))
        fig.update_layout(title="ROC curve (hold-out)", xaxis_title="False positive rate", yaxis_title="True positive rate", height=360)
        st.plotly_chart(fig, width="stretch")
    with r_:
        cm = clf["confusion"]
        fig = px.imshow(cm, text_auto=True, x=["Pred: not interested", "Pred: interested"], y=["Actual: not", "Actual: interested"],
                        color_continuous_scale="Greys", title="Confusion matrix (hold-out)")
        fig.update_layout(height=360, coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")
    imp = clf["importance"].head(12)[::-1]
    fig = px.bar(x=imp.values, y=imp.index, orientation="h", title="What drives interest (permutation importance, AUC drop)",
                 color_discrete_sequence=["#3a3a3a"])
    fig.update_layout(height=420, xaxis_title="Importance", yaxis_title="")
    st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ Clustering
with tabs[3]:
    st.subheader("Clustering - High / Medium / Low conversion segments")
    st.caption("K-means on attitudes, income and spend (no intent answers). Clusters are then ranked by the classifier's out-of-sample probability of converting.")
    s = clu["summary"]
    show = s.rename(columns={"size": "Respondents", "share": "Share", "avg_conversion_prob": "Avg conv. prob", "actually_interested": "% interested (actual)",
                             "income": "Median income", "monthly_clothing_spend": "Median monthly clothing AED", "wtp_tshirt": "Median WTP tee AED",
                             "grey_pref": "Grey pref (1-5)", "sustain": "Sustain. (1-5)", "loyalty": "Loyalty (1-5)", "appeal": "Concept appeal (1-5)",
                             "spend_6m": "Median 6-mo spend AED"})
    fmt = {c: "{:.0%}" for c in ["Share", "Avg conv. prob", "% interested (actual)"]}
    fmt.update({c: "{:,.0f}" for c in ["Median income", "Median monthly clothing AED", "Median WTP tee AED", "Median 6-mo spend AED"]})
    fmt.update({c: "{:.2f}" for c in ["Grey pref (1-5)", "Sustain. (1-5)", "Loyalty (1-5)", "Concept appeal (1-5)"]})
    st.dataframe(show.style.format(fmt), width="stretch")
    l, r_ = st.columns(2)
    with l:
        fig = px.scatter(clu["pca"], x="PC1", y="PC2", color="segment", color_discrete_map=SEG_COLORS, opacity=0.8,
                         title="Segments in 2-D (PCA)")
        fig.update_layout(height=380)
        st.plotly_chart(fig, width="stretch")
    with r_:
        sil = pd.Series(clu["silhouette"])
        fig = px.line(x=sil.index, y=sil.values, markers=True, title="Silhouette score by number of clusters (higher = cleaner)",
                      color_discrete_sequence=["#3a3a3a"])
        fig.update_layout(height=380, xaxis_title="k", yaxis_title="Silhouette")
        st.plotly_chart(fig, width="stretch")
        st.caption("Segments overlap (a normal result for attitudes data). k=3 is chosen for business reasons - three actionable groups - not because it is mathematically perfect.")
    dim = st.selectbox("Who are they?", ["q01_age_group", "q02_gender", "q08_primary_brand", "q28_discovery"],
                       format_func=lambda x: {"q01_age_group": "Age group", "q02_gender": "Gender", "q08_primary_brand": "Current brand",
                                              "q28_discovery": "Where they would discover us"}[x])
    ct = clu["demographics"][dim].reset_index().melt("segment")
    fig = px.bar(ct, x="segment", y="value", color=ct.columns[1], barmode="stack", color_discrete_sequence=px.colors.sequential.Greys_r[:6] + ["#9b2c2c"],
                 category_orders={"segment": ["High probability", "Medium probability", "Low probability"]})
    fig.update_layout(height=340, yaxis_tickformat=".0%", yaxis_title="Share of segment", xaxis_title="")
    st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ Association
with tabs[4]:
    st.subheader("Association rules - what to launch first")
    st.caption(f"Mined on {asr['n_base']} respondents with intent ≥ 3 (real buyers, not everyone). "
               "Support = how common; Confidence = if A then B, how often; Lift > 1 = more than chance.")
    cv = asr["curve"]
    l, r_ = st.columns(2)
    with l:
        fig = go.Figure()
        fig.add_scatter(x=cv["launch_size"], y=cv["interest_covered"], name="Interest covered", mode="lines+markers", line_color="#1f1f1f")
        fig.add_scatter(x=cv["launch_size"], y=cv["reach_any"], name="Reach (≥1 wanted item available)", mode="lines+markers", line_color="#9a9a9a")
        fig.add_scatter(x=cv["launch_size"], y=cv["full_basket_satisfied"], name="Full basket available", mode="lines+markers", line_color="#9b2c2c")
        fig.add_hline(y=0.75, line_dash="dash", line_color="#bbb")
        fig.update_layout(title="Launch with N categories (greedy order)", xaxis_title="Categories at launch", yaxis_tickformat=".0%", height=380)
        st.plotly_chart(fig, width="stretch")
    with r_:
        sup = asr["item_support"][::-1]
        fig = px.bar(x=sup.values, y=sup.index, orientation="h", title="Demand per item", color_discrete_sequence=["#3a3a3a"])
        fig.update_layout(height=380, xaxis_tickformat=".0%", yaxis_title="", xaxis_title="Share of respondents")
        st.plotly_chart(fig, width="stretch")
    order = [ITEMS[c] for c in asr["order"]]
    k = st.slider("Launch categories", 1, 8, asr["recommended_k"])
    st.info(f"**Launch set ({k}):** " + ", ".join(order[:k]) + f".  Holds back: {', '.join(order[k:]) or 'nothing'}.  "
            f"Interest covered: **{cv.loc[k - 1, 'interest_covered']:.0%}**, full basket available: **{cv.loc[k - 1, 'full_basket_satisfied']:.0%}**.")
    st.markdown("**Rule explorer**")
    c1, c2, c3 = st.columns(3)
    ms = c1.slider("Min support", 0.05, 0.40, 0.10, 0.01)
    mc = c2.slider("Min confidence", 0.3, 0.95, 0.60, 0.05)
    ml = c3.slider("Min lift", 1.0, 2.0, 1.10, 0.05)
    from src.analysis import run_association
    res = run_association(d, ms, mc, ml, launch_k=k)
    st.dataframe(res["rules"].head(40).style.format({"support": "{:.0%}", "confidence": "{:.0%}", "lift": "{:.2f}"}), hide_index=True, width="stretch")
    st.markdown("**⚠ The 'missing C' risk** - rules where the customer wants launched items *and* a held-back category:")
    g = res["gaps"]
    st.dataframe(g.head(10).style.format({"support": "{:.0%}", "confidence": "{:.0%}", "lift": "{:.2f}"}) if len(g)
                 else pd.DataFrame({"": ["No strong gap rules at this setting"]}), hide_index=True, width="stretch")

# ------------------------------------------------------------------ Regression
with tabs[5]:
    st.subheader("Regression - how much will they spend in 6 months?")
    st.caption("Target: Q25 (AED, log-transformed to tame outliers). Q26/Q27/NPS excluded.")
    st.dataframe(reg["metrics"].style.format("{:.3f}"), width="stretch")
    st.success(f"Best model: **{reg['best']}**. R² is on log-AED; the AED error columns are what a budget owner cares about.")
    l, r_ = st.columns(2)
    with l:
        fig = px.scatter(x=reg["actual"], y=reg["predicted"], opacity=0.5, labels={"x": "Actual AED (Q25)", "y": "Predicted AED (out-of-fold)"},
                         color_discrete_sequence=["#3a3a3a"], title="Predicted vs actual")
        mx = float(max(np.nanmax(reg["actual"]), np.nanmax(reg["predicted"])))
        fig.add_scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(dash="dash", color="#bbb"), name="Perfect")
        fig.update_layout(height=380)
        st.plotly_chart(fig, width="stretch")
    with r_:
        imp = reg["importance"].head(10)[::-1]
        fig = px.bar(x=imp.values, y=imp.index, orientation="h", title="What drives spend (permutation importance)", color_discrete_sequence=["#3a3a3a"])
        fig.update_layout(height=380, yaxis_title="", xaxis_title="R² drop")
        st.plotly_chart(fig, width="stretch")
    pr = d.assign(pred=reg["pred_all"], segment=clu["segments"])
    seg_sp = pr.groupby("segment")["pred"].agg(["median", "mean"]).reindex(["High probability", "Medium probability", "Low probability"]).round(0)
    st.markdown("**Predicted 6-month spend per segment (AED)**")
    st.dataframe(seg_sp.rename(columns={"median": "Median", "mean": "Mean"}), width="stretch")

# ------------------------------------------------------------------ Unit economics
with tabs[6]:
    st.subheader("Unit economics & stress test")
    b, t = UE["broad"], UE["targeted"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Real annual spend / customer", f"AED {UE['annual_spend']:,.0f}")
    c2.metric("Contribution margin", f"{UE['cm']:.0%}")
    c3.metric("Customer lifetime value", f"AED {UE['ltv']:,.0f}")
    c4.metric("Break-even customers (broad)", f"{b['breakeven_customers']:,.0f}" if np.isfinite(b["breakeven_customers"]) else "never")
    cmp_ = pd.DataFrame({"Broad campaign": b, "Model-targeted (High segment)": t}).loc[
        ["customers", "revenue", "contribution", "marketing", "fixed", "profit_y1", "cac", "ltv_cac"]]
    cmp_.index = ["Customers won", "Revenue (AED)", "Contribution (AED)", "Marketing spend (AED)", "Fixed costs (AED)", "Year-1 profit (AED)", "CAC (AED)", "LTV : CAC"]
    st.dataframe(cmp_.style.format("{:,.1f}"), width="stretch")
    st.caption("Targeted CAC = broad CAC × (overall interest rate ÷ High-segment interest rate): you only pay to reach people the model scores highly, "
               "but you can only win customers inside that smaller pool.")
    st.markdown("**Stress test - Year-1 profit (AED) by intent→purchase rate and CAC**")
    rates = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
    cacs = [50, 70, 90, 120, 150, 200]
    grid = pd.DataFrame([[unit_economics(R, Assumptions(**{**A.as_dict(), "intent_to_purchase": r, "cac": c}))["broad"]["profit_y1"]
                          for r in rates] for c in cacs], index=[f"CAC {c}" for c in cacs], columns=[f"{int(r * 100)}%" for r in rates])
    fig = px.imshow(grid / 1e6, text_auto=".1f", color_continuous_scale=["#9b2c2c", "#ffffff", "#3a3a3a"], color_continuous_midpoint=0,
                    labels=dict(color="AED m"), aspect="auto")
    fig.update_layout(height=360, xaxis_title="Share of 'interested' who really buy", yaxis_title="")
    st.plotly_chart(fig, width="stretch")
    st.caption("Red cells = losing money in year 1. The idea lives or dies on the real conversion rate and CAC - exactly what the real survey and a test ad campaign must measure.")

# ------------------------------------------------------------------ Strategy
with tabs[7]:
    st.subheader("Go-to-market strategy generated from the results")
    hi, med, lo = [clu["summary"].loc[n] for n in ["High probability", "Medium probability", "Low probability"]]
    pr = d.assign(segment=clu["segments"])
    hi_df = pr[pr["segment"] == "High probability"]
    top_age = hi_df["q01_age_group"].value_counts(normalize=True)
    top_ch = hi_df["q28_discovery"].value_counts(normalize=True)
    popup = (hi_df["q23_popup"] == "Yes").mean()
    sub_yes = (hi_df["q22_subscribe"] == "Yes").mean()
    sub_any = hi_df["q22_subscribe"].isin(["Yes", "Maybe"]).mean()
    wtp_hi = hi_df["q15_wtp_tshirt_aed"].median()
    order = [ITEMS[c] for c in asr["order"]]
    st.markdown(f"""
**1. Who to target** - the *High-probability* segment ({hi['share']:.0%} of respondents, {hi['actually_interested']:.0%} interested).
Mostly {top_age.index[0]} ({top_age.iloc[0]:.0%}); median concept appeal {hi['appeal']:.1f}/5, grey preference {hi['grey_pref']:.1f}/5, lower brand loyalty ({hi['loyalty']:.1f}/5).

**2. Who not to chase with paid ads** - the *Low-probability* segment ({lo['share']:.0%}; only {lo['actually_interested']:.0%} interested) is loyal to current brands
(loyalty {lo['loyalty']:.1f}/5) even with decent income. Reach them later via retargeting or pop-ups, not acquisition spend.

**3. Nurture** - the *Medium* segment ({med['share']:.0%}) has low income (median AED {med['income']:,.0f}). Entry-price basics and limited drops, no premium upsell.

**4. What to launch** - start with **{asr['recommended_k']} categories: {', '.join(order[:asr['recommended_k']])}**
(covers {cv.loc[asr['recommended_k'] - 1, 'interest_covered']:.0%} of expressed interest). Hold back: {', '.join(order[asr['recommended_k']:]) or 'none'}.

**5. Price point** - the High segment's median max price for a tee is **AED {wtp_hi:,.0f}** versus your planned AED {A.planned_tshirt_price:,.0f}.
Premium tolerance: {(d['q16_premium_pct'].fillna(0) >= 10).mean():.0%} would pay ≥10% extra for sustainable fabric - keep the eco premium small and prove it with certification.

**6. Channels** - top discovery channel in the High segment: **{top_ch.index[0]}** ({top_ch.iloc[0]:.0%}), then {top_ch.index[1]} ({top_ch.iloc[1]:.0%}).
{popup:.0%} would attend a pop-up, so use pop-ups for trial and community, not volume.

**7. Subscription box** - {sub_yes:.0%} say yes and {sub_any:.0%} yes/maybe in the High segment. {'Pilot with a small waitlist first.' if sub_yes < 0.30 else 'Strong enough to test as a core revenue stream.'}

**8. Biggest risk to test with real data** - grey-only is a narrow assortment: check repeat-purchase intent and the 'only grey is too limiting' objection (Q30 text → sentiment analysis, session 8).
""")
    objections = d["q30_open_text"].value_counts().head(6)
    st.markdown("**Top stated objections / comments (Q30)**")
    st.dataframe(objections.rename("count"), width="stretch")

# ------------------------------------------------------------------ Survey
with tabs[8]:
    st.subheader("30-question survey, aligned to the algorithms")
    sv = pd.DataFrame(SURVEY, columns=["Q#", "Section", "Question", "Answer type", "Data column", "Used by"])
    algo = st.multiselect("Filter by algorithm", ["Classification", "Clustering", "Association", "Regression", "Pricing", "Text analytics"])
    view = sv[sv["Used by"].str.contains("|".join(algo))] if algo else sv
    st.dataframe(view, hide_index=True, width="stretch", height=620)
    st.download_button("⬇ Download questionnaire (CSV)", sv.to_csv(index=False), "questionnaire.csv", "text/csv")
