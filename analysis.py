"""Cleaning + the four algorithms: classification, clustering, association rules, regression."""
from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             precision_score, r2_score, recall_score, roc_auc_score, roc_curve, silhouette_score)
from sklearn.model_selection import cross_val_predict, cross_val_score, train_test_split, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .schema import (CAT_COLS, ITEMS, LEAKY_FOR_CLASSIFICATION, LEAKY_FOR_REGRESSION, LIKERT,
                     NUMERIC_OUTLIER_COLS, TARGET_CLASS, TARGET_REG)

SEED = 42


# ------------------------------------------------------------------ cleaning
def clean(df):
    """Dedupe, drop careless responders, cap outliers (IQR winsorising), tag target. Returns (df, report)."""
    raw = df.copy()
    rep = {"rows_raw": len(raw)}
    d = raw.drop_duplicates(subset=[c for c in raw.columns if c != "respondent_id"]).copy()
    rep["duplicates_removed"] = len(raw) - len(d)

    lk = [c for c in LIKERT if c in d.columns and c != "q21_purchase_intent"]
    careless = (d[lk].nunique(axis=1) == 1) & (d[lk].iloc[:, 0] == 3)
    rep["careless_removed"] = int(careless.sum())
    d = d[~careless].copy()

    rep["missing_cells"] = int(d.isna().sum().sum())
    rep["outliers_capped"] = {}
    for c in [c for c in NUMERIC_OUTLIER_COLS if c in d.columns]:
        s = d[c]
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        hi = q3 + 3 * (q3 - q1)  # 3xIQR = "extreme" outliers only, real big spenders stay
        n_out = int((s > hi).sum())
        rep["outliers_capped"][c] = n_out
        d[c] = s.clip(upper=hi)
    d[TARGET_CLASS] = (d["q21_purchase_intent"] >= 4).astype(int)
    rep["rows_clean"] = len(d)
    rep["raw_numeric_describe"] = raw[[c for c in NUMERIC_OUTLIER_COLS if c in raw.columns]].describe().T
    return d.reset_index(drop=True), rep


def _split_cols(d, drop):
    cols = [c for c in d.columns if c not in drop + ["respondent_id"]]
    cat = [c for c in cols if not pd.api.types.is_numeric_dtype(d[c])]
    num = [c for c in cols if c not in cat]
    return num, cat


def _prep(num, cat):
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat)])


# ------------------------------------------------------------------ 1. classification
def run_classification(d):
    num, cat = _split_cols(d, LEAKY_FOR_CLASSIFICATION)
    X, y = d[num + cat], d[TARGET_CLASS]
    models = {
        "Logistic Regression": Pipeline([("p", _prep(num, cat)), ("m", LogisticRegression(max_iter=2000, C=0.5))]),
        "Random Forest": Pipeline([("p", _prep(num, cat)),
                                   ("m", RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=SEED))]),
    }
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rows, fitted, roc = [], {}, {}
    for name, m in models.items():
        cv_auc = cross_val_score(m, X, y, cv=skf, scoring="roc_auc")
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        pred = (p >= 0.5).astype(int)
        rows.append({"Model": name, "CV AUC (5-fold)": cv_auc.mean(), "Test AUC": roc_auc_score(yte, p),
                     "Accuracy": accuracy_score(yte, pred), "Precision": precision_score(yte, pred),
                     "Recall": recall_score(yte, pred), "F1": f1_score(yte, pred)})
        fitted[name] = (m, p, pred)
        fpr, tpr, _ = roc_curve(yte, p)
        roc[name] = (fpr, tpr)
    metrics = pd.DataFrame(rows).set_index("Model")
    best = metrics["CV AUC (5-fold)"].idxmax()
    m, p, pred = fitted[best]
    imp = permutation_importance(m, Xte, yte, scoring="roc_auc", n_repeats=15, random_state=SEED)
    importance = pd.Series(imp.importances_mean, index=num + cat).sort_values(ascending=False)
    # out-of-fold probabilities for every respondent (no leakage) -> used by clustering/viability
    prob = cross_val_predict(models[best], X, y, cv=skf, method="predict_proba")[:, 1]
    return {"metrics": metrics, "best": best, "confusion": confusion_matrix(yte, pred), "roc": roc,
            "importance": importance, "prob": prob, "base_rate": float(y.mean()), "features": num + cat}


# ------------------------------------------------------------------ 2. clustering
CLUSTER_FEATURES = ["q05_income_aed", "q07_clothing_spend_aed", "q10_grey_pref", "q11_sustain_importance",
                    "q12_brand_loyalty", "q13_concept_appeal", "q14_price_fairness", "q15_wtp_tshirt_aed",
                    "q17_switch_likelihood", "q29_influencer", "q06_orders_per_month"]


def run_clustering(d, prob, k=3):
    X = d[CLUSTER_FEATURES].copy()
    for c in ["q05_income_aed", "q07_clothing_spend_aed", "q15_wtp_tshirt_aed"]:
        X[c] = np.log1p(X[c])
    Xs = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(X))
    sil = {kk: silhouette_score(Xs, KMeans(kk, n_init=10, random_state=SEED).fit_predict(Xs)) for kk in range(2, 7)}
    km = KMeans(k, n_init=20, random_state=SEED).fit(Xs)
    lab = pd.Series(km.labels_, index=d.index)
    order = pd.Series(prob, index=d.index).groupby(lab).mean().sort_values(ascending=False).index
    names = {order[0]: "High probability", order[1]: "Medium probability", order[2]: "Low probability"} if k == 3 else \
            {c: f"Segment {i + 1}" for i, c in enumerate(order)}
    seg = lab.map(names)
    prof = d.assign(segment=seg, p_convert=prob)
    summary = prof.groupby("segment").agg(
        size=("segment", "size"), avg_conversion_prob=("p_convert", "mean"), actually_interested=(TARGET_CLASS, "mean"),
        income=("q05_income_aed", "median"), monthly_clothing_spend=("q07_clothing_spend_aed", "median"),
        wtp_tshirt=("q15_wtp_tshirt_aed", "median"), grey_pref=("q10_grey_pref", "mean"),
        sustain=("q11_sustain_importance", "mean"), loyalty=("q12_brand_loyalty", "mean"),
        appeal=("q13_concept_appeal", "mean"), spend_6m=("q25_spend_6m_aed", "median"))
    summary["share"] = summary["size"] / summary["size"].sum()
    summary = summary.reindex([n for n in ["High probability", "Medium probability", "Low probability"] if n in summary.index]
                              or summary.index)
    demo = {c: pd.crosstab(prof["segment"], prof[c], normalize="index") for c in ["q01_age_group", "q02_gender", "q08_primary_brand", "q28_discovery"]}
    pc = PCA(2, random_state=SEED).fit_transform(Xs)
    return {"segments": seg, "summary": summary, "silhouette": sil, "pca": pd.DataFrame(pc, columns=["PC1", "PC2"]).assign(segment=seg.values),
            "demographics": demo}


# ------------------------------------------------------------------ 3. association rules (own Apriori, no extra dependency)
def run_association(d, min_support=0.10, min_conf=0.60, min_lift=1.10, max_len=3, base="interested", launch_k=None):
    """Rules are mined on respondents who show real interest (intent >= 3) so they reflect buyers, not everybody."""
    sub = d[d["q21_purchase_intent"] >= 3] if base == "interested" else d
    B = sub[list(ITEMS)].fillna(0).astype(bool)
    n = len(B)
    labels = ITEMS
    sup = {}
    for r in range(1, max_len + 1):
        for combo in combinations(B.columns, r):
            s = B[list(combo)].all(axis=1).mean()
            if s >= min_support:
                sup[frozenset(combo)] = s
    rules = []
    for items, s in sup.items():
        if len(items) < 2:
            continue
        for r in range(1, len(items)):
            for ante in combinations(items, r):
                ante = frozenset(ante)
                cons = items - ante
                if ante in sup and cons in sup:
                    conf, lift = s / sup[ante], s / (sup[ante] * sup[cons])
                    if conf >= min_conf and lift >= min_lift:
                        rules.append({"If they want": " + ".join(labels[a] for a in sorted(ante)),
                                      "Then they also want": " + ".join(labels[c] for c in sorted(cons)),
                                      "support": s, "confidence": conf, "lift": lift,
                                      "_ante": ante, "_cons": cons})
    rules = pd.DataFrame(rules).sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True) if rules else \
        pd.DataFrame(columns=["If they want", "Then they also want", "support", "confidence", "lift", "_ante", "_cons"])
    item_support = B.mean().rename(index=labels).sort_values(ascending=False)

    # Launch-assortment planner: greedy pick of categories to maximise share of respondent interest covered
    cat_cols = [c for c in ITEMS if c.startswith("q19_")]
    Bc = B[cat_cols]
    total = Bc.values.sum()
    chosen, curve = [], []
    for _ in cat_cols:
        best = max((c for c in cat_cols if c not in chosen), key=lambda c: Bc[chosen + [c]].values.sum())
        chosen.append(best)
        covered = Bc[chosen].values.sum() / total
        reach = Bc[chosen].any(axis=1).mean()
        # "full basket" = everything the respondent wants is available
        full = (Bc.drop(columns=chosen).sum(axis=1) == 0).mean()
        curve.append({"launch_size": len(chosen), "added": labels[best], "interest_covered": covered,
                      "reach_any": reach, "full_basket_satisfied": full})
    curve = pd.DataFrame(curve)
    # "missing C" risk: rules whose antecedent is launched but consequent is not
    def gaps(k):
        S = {c for c in chosen[:k]}
        if rules.empty:
            return rules
        g = rules[rules["_ante"].apply(lambda a: a <= S) & rules["_cons"].apply(lambda c: not (c <= S) and all(x.startswith("q19_") for x in c))]
        return g.drop(columns=["_ante", "_cons"])
    ok = curve[curve["interest_covered"] >= 0.75]
    rec_k = int(ok["launch_size"].iloc[0]) if len(ok) else len(chosen)
    return {"rules": rules.drop(columns=["_ante", "_cons"]), "item_support": item_support, "curve": curve,
            "order": chosen, "recommended_k": rec_k, "gaps": gaps(launch_k or rec_k), "n_base": n, "base_kind": base}


# ------------------------------------------------------------------ 4. regression
def run_regression(d):
    sub = d[d[TARGET_REG].notna()].copy()
    num, cat = _split_cols(sub, LEAKY_FOR_REGRESSION + [TARGET_REG])
    X, y = sub[num + cat], np.log1p(sub[TARGET_REG])
    kf = KFold(5, shuffle=True, random_state=SEED)
    models = {"Ridge": Pipeline([("p", _prep(num, cat)), ("m", Ridge(alpha=5.0))]),
              "Gradient Boosting": Pipeline([("p", _prep(num, cat)),
                                             ("m", GradientBoostingRegressor(n_estimators=250, max_depth=3, learning_rate=0.05,
                                                                              subsample=0.8, random_state=SEED))])}
    rows = []
    for name, m in models.items():
        r2 = cross_val_score(m, X, y, cv=kf, scoring="r2").mean()
        pred = np.expm1(cross_val_predict(m, X, y, cv=kf))
        rows.append({"Model": name, "CV R2 (log AED)": r2, "MAE (AED)": mean_absolute_error(sub[TARGET_REG], pred),
                     "Median abs error (AED)": float(np.median(np.abs(sub[TARGET_REG] - pred)))})
    metrics = pd.DataFrame(rows).set_index("Model")
    best = metrics["CV R2 (log AED)"].idxmax()
    final = models[best].fit(X, y)
    oof = np.expm1(cross_val_predict(models[best], X, y, cv=kf))
    ridge = models["Ridge"].fit(X, y)
    names = ridge.named_steps["p"].get_feature_names_out()
    coef = pd.Series(ridge.named_steps["m"].coef_, index=[n.split("__", 1)[1] for n in names]).sort_values(key=abs, ascending=False)
    imp = permutation_importance(final, X, y, scoring="r2", n_repeats=10, random_state=SEED)
    importance = pd.Series(imp.importances_mean, index=num + cat).sort_values(ascending=False)
    full_pred = pd.Series(np.expm1(final.predict(d[num + cat])), index=d.index)
    return {"metrics": metrics, "best": best, "actual": sub[TARGET_REG].values, "predicted": oof,
            "coef": coef, "importance": importance, "pred_all": full_pred, "model": final, "features": num + cat}


def run_all(raw_df, k=3):
    d, rep = clean(raw_df)
    clf = run_classification(d)
    clu = run_clustering(d, clf["prob"], k=k)
    asr = run_association(d)
    reg = run_regression(d)
    return {"data": d, "report": rep, "clf": clf, "clu": clu, "asr": asr, "reg": reg}
