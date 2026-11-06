"""Synthetic survey generator for "Shades of Grey".

Produces a realistic, *deliberately messy* dataset that follows the 30-question survey:
  - latent traits drive the answers, so there are real patterns for the 4 algorithms to find
  - intentional noise: label flips, careless respondents, missing values, duplicates
  - intentional outliers: typo-style (extra zero) and extreme high spenders

IMPORTANT: every pattern here is an ASSUMPTION written by us. The model output on this data
tests that the pipeline works; it does NOT prove the market wants the product. Replace with
real survey data (same columns) as soon as it exists.
"""
import numpy as np
import pandas as pd
from .schema import CATEGORIES, FEATURES

BRANDS = ["H&M", "Zara", "Uniqlo", "Shein", "Local/Other"]
DISCOVERY = ["Instagram", "TikTok", "Facebook", "Friends", "Pop-up", "Search"]


def _sig(x):
    return 1 / (1 + np.exp(-x))


def _likert(rng, mean, sd=0.9):
    return np.clip(np.rint(mean + rng.normal(0, sd, size=np.shape(mean))), 1, 5).astype(int)


def generate(n=600, seed=42, noise=1.0):
    """noise=0 -> clean data, 1 -> default messy, 2 -> very messy."""
    rng = np.random.default_rng(seed)

    # ---- Profile (Q1-Q6)
    age_group = rng.choice(["16-24", "25-34", "35-50"], n, p=[0.34, 0.40, 0.26])
    gender = rng.choice(["Female", "Male", "Prefer not to say"], n, p=[0.52, 0.45, 0.03])
    residence = rng.choice(["Metro core", "Suburban", "Other"], n, p=[0.62, 0.28, 0.10])
    occupation = np.where(age_group == "16-24",
                          rng.choice(["Student", "Employed", "Self-employed", "Other"], n, p=[0.62, 0.28, 0.06, 0.04]),
                          rng.choice(["Student", "Employed", "Self-employed", "Other"], n, p=[0.04, 0.74, 0.17, 0.05]))
    inc_med = np.select([occupation == "Student", occupation == "Employed", occupation == "Self-employed"],
                        [2500, 11000, 14000], 6000) * np.where(age_group == "35-50", 1.35, 1.0)
    income = np.round(inc_med * rng.lognormal(0, 0.45, n), -2)
    orders_pm = np.clip(rng.poisson(np.where(residence == "Metro core", 3.0, 2.0)), 0, 20)

    # ---- Latent traits
    eco = rng.normal(0, 1, n) + np.where(age_group == "16-24", 0.2, 0) + np.where(gender == "Female", 0.15, 0)
    neutral = rng.normal(0, 1, n) + np.where(occupation == "Employed", 0.25, 0) + np.where(age_group == "35-50", 0.2, 0)
    price_sens = rng.normal(0, 1, n) - 0.35 * (np.log(income) - np.log(income).mean()) / np.log(income).std()
    social = rng.normal(0, 1, n) + np.where(age_group == "16-24", 0.6, np.where(age_group == "25-34", 0.1, -0.4))
    loyalty_t = rng.normal(0, 1, n)

    # ---- Current behaviour (Q7-Q12)
    spend_m = np.clip(income * rng.uniform(0.03, 0.09, n) * np.exp(0.12 * social) + rng.normal(0, 40, n), 40, None)
    spend_m = np.round(spend_m)
    brand_p = np.stack([0.30 + 0.05 * (price_sens > 0), 0.22 + 0.06 * (income > 12000), 0.16 + 0.10 * (neutral > 0.5),
                        0.18 + 0.15 * (price_sens > 0.8), 0.14 * np.ones(n)], 1)
    brand_p = brand_p / brand_p.sum(1, keepdims=True)
    brand = np.array([rng.choice(BRANDS, p=p) for p in brand_p])
    avg_item = np.clip(np.rint(55 + 30 * (np.log(income) - np.log(income).mean()) + rng.normal(0, 15, n)
                               + np.select([brand == "Shein", brand == "Zara", brand == "Uniqlo"], [-22, 12, 8], 0)), 15, None)
    grey_pref = _likert(rng, 3 + 0.75 * neutral)
    sustain = _likert(rng, 3 + 0.75 * eco)
    loyalty = _likert(rng, 3 + 0.6 * loyalty_t + 0.15 * (brand == "Uniqlo"))

    # ---- Concept test (Q13-Q18)
    appeal = _likert(rng, 3 + 0.55 * neutral + 0.25 * eco + 0.15 * social, 0.8)
    fairness = _likert(rng, 3 - 0.5 * price_sens + 0.15 * eco, 0.9)
    wtp = np.clip(np.rint(avg_item * 0.85 * np.exp(0.18 * eco + 0.12 * neutral - 0.12 * price_sens) + rng.normal(0, 8, n)), 10, None)
    prem = np.clip(np.rint((5 * np.rint(np.clip(0.35 * (eco + 0.6) - 0.15 * price_sens + rng.normal(0.65, 0.9, n), 0, 5)))), 0, 25).astype(int)
    switch = _likert(rng, 3 + 0.4 * (appeal - 3) - 0.45 * (loyalty - 3) + 0.2 * social)
    trust = _likert(rng, 3 + 0.35 * eco + 0.1 * (sustain - 3))

    # ---- Latent conversion propensity (what the classifier should discover)
    z = (0.60 * (appeal - 3) + 0.38 * (grey_pref - 3) + 0.27 * (sustain - 3) + 0.34 * (fairness - 3)
         + 0.32 * (switch - 3) - 0.18 * (loyalty - 3) + 0.12 * (trust - 3) + 0.15 * social
         + 0.0000 * income + rng.normal(0, 0.85, n) - 0.55)
    p_int = _sig(z)
    intent = np.clip(np.rint(1 + 4 * p_int + rng.normal(0, 0.45, n)), 1, 5).astype(int)

    # ---- Product interest (Q19-Q20): correlated through shared style factors
    casual = 0.5 * social + 0.6 * z + rng.normal(0, 0.7, n)
    formal = 0.5 * (occupation == "Employed") + 0.3 * neutral + 0.4 * z + rng.normal(0, 0.8, n)
    cat_logit = {
        "q19_tshirts": 1.5 + 0.6 * casual, "q19_hoodies": 0.6 + 0.8 * casual + 0.4 * (age_group == "16-24"),
        "q19_trousers": 0.7 + 0.5 * casual + 0.3 * formal, "q19_outerwear": -0.3 + 0.4 * casual + 0.4 * formal,
        "q19_formal": -0.9 + 0.9 * formal, "q19_knitwear": -0.7 + 0.5 * formal + 0.4 * eco,
        "q19_activewear": -0.9 + 0.5 * casual + 0.5 * (age_group == "16-24"), "q19_accessories": -0.4 + 0.4 * z + 0.2 * social,
    }
    feat_logit = {
        "q20_subscription_box": -0.9 + 0.7 * z + 0.3 * social, "q20_limited_editions": -0.6 + 0.6 * social + 0.4 * z,
        "q20_eco_certified": -0.2 + 0.9 * eco, "q20_styling_guide": -0.3 + 0.5 * neutral + 0.3 * z,
        "q20_loyalty": 0.1 + 0.3 * price_sens, "q20_free_returns": 0.7 + 0.2 * price_sens, "q20_custom_fit": -1.2 + 0.4 * formal,
    }
    items = {k: (rng.random(n) < _sig(v)).astype(int) for k, v in {**cat_logit, **feat_logit}.items()}

    # ---- Conversion & spend (Q21-Q27)
    sub_p = np.where(items["q20_subscription_box"] == 1, _sig(z - 0.2), _sig(z - 1.8))
    subscribe = np.array(["Yes" if r < p * 0.55 else ("Maybe" if r < p else "No") for r, p in zip(rng.random(n), sub_p)])
    popup = np.where(rng.random(n) < _sig(0.7 * z + 0.4 * social - 0.2), "Yes", "No")
    nps = np.clip(np.rint(6 + 2.2 * z + rng.normal(0, 1.4, n)), 0, 10).astype(int)
    share = _sig(1.6 * z - 0.9) * 0.30
    spend_6m = np.round(spend_m * 6 * share * rng.lognormal(0, 0.28, n) * (1 + 0.15 * (wtp > 60)))
    orders_py = np.clip(np.rint(1 + 7 * _sig(z) * rng.lognormal(0, 0.3, n)), 0, None).astype(int)
    sub_max = np.clip(np.rint(avg_item * 0.9 * _sig(z) * rng.lognormal(0, 0.25, n)), 0, None)
    one = np.ones(n)
    p_disc = np.stack([0.20 + 0.20 * (age_group == "16-24"), 0.08 + 0.28 * (age_group == "16-24"), 0.10 * one,
                       0.22 * one, 0.12 * one, 0.14 * one], 1)
    p_disc /= p_disc.sum(1, keepdims=True)
    discovery = np.array([rng.choice(DISCOVERY, p=p) for p in p_disc])
    influencer = _likert(rng, 3 + 0.5 * social)

    # ---- Open text (Q30) for the later sentiment homework
    neg = ["Only grey is too limiting, I need colour", "Worried the quality will not justify the price",
           "I already shop at H&M and Zara, why switch", "Delivery and returns need to be easy",
           "Not sure the sustainable claim is genuine", "Too expensive compared to Shein"]
    neu = ["Depends on the fit and fabric", "Would want to try in a pop-up first", "Need more category choice"]
    pos = ["Nothing really, I love the minimalist idea", "Great concept, please add more basics",
           "Love eco-friendly fabrics, would buy immediately"]
    text = np.array([rng.choice(pos if i >= 4 else neu if i == 3 else neg) for i in intent])

    df = pd.DataFrame({
        "q01_age_group": age_group, "q02_gender": gender, "q03_residence": residence, "q04_occupation": occupation,
        "q05_income_aed": income, "q06_orders_per_month": orders_pm, "q07_clothing_spend_aed": spend_m,
        "q08_primary_brand": brand, "q09_avg_item_price_aed": avg_item, "q10_grey_pref": grey_pref,
        "q11_sustain_importance": sustain, "q12_brand_loyalty": loyalty, "q13_concept_appeal": appeal,
        "q14_price_fairness": fairness, "q15_wtp_tshirt_aed": wtp, "q16_premium_pct": prem,
        "q17_switch_likelihood": switch, "q18_trust_claims": trust, **items,
        "q21_purchase_intent": intent, "q22_subscribe": subscribe, "q23_popup": popup, "q24_nps": nps,
        "q25_spend_6m_aed": spend_6m, "q26_orders_per_year": orders_py, "q27_sub_max_aed": sub_max,
        "q28_discovery": discovery, "q29_influencer": influencer, "q30_open_text": text,
    })

    # ================= Intentional noise & outliers =================
    if noise > 0:
        k = lambda frac: int(round(n * frac * noise))
        # 1) label noise: respondents who say 1-2 points differently from their true propensity
        idx = rng.choice(n, k(0.06), replace=False)
        df.loc[idx, "q21_purchase_intent"] = np.clip(df.loc[idx, "q21_purchase_intent"] + rng.choice([-2, -1, 1, 2], len(idx)), 1, 5)
        # 2) careless respondents (straight-lining all 3s)
        idx = rng.choice(n, k(0.03), replace=False)
        for c in ["q10_grey_pref", "q11_sustain_importance", "q12_brand_loyalty", "q13_concept_appeal",
                  "q14_price_fairness", "q17_switch_likelihood", "q18_trust_claims", "q29_influencer"]:
            df.loc[idx, c] = 3
        # 3) typo-style outliers (extra zero) and extreme spenders
        for col, frac, mult in [("q07_clothing_spend_aed", 0.015, 10), ("q05_income_aed", 0.012, 10),
                                ("q15_wtp_tshirt_aed", 0.012, 10), ("q09_avg_item_price_aed", 0.01, 10)]:
            idx = rng.choice(n, k(frac), replace=False)
            df.loc[idx, col] = df.loc[idx, col] * mult
        idx = rng.choice(n, k(0.02), replace=False)
        df.loc[idx, "q25_spend_6m_aed"] = df.loc[idx, "q25_spend_6m_aed"] * rng.uniform(5, 12, len(idx))
        idx = rng.choice(n, k(0.01), replace=False)
        df.loc[idx, "q06_orders_per_month"] = rng.integers(40, 90, len(idx))
        # 4) missing values
        for col in ["q05_income_aed", "q07_clothing_spend_aed", "q09_avg_item_price_aed", "q15_wtp_tshirt_aed",
                    "q16_premium_pct", "q10_grey_pref", "q25_spend_6m_aed", "q08_primary_brand"]:
            df.loc[rng.choice(n, k(0.03), replace=False), col] = np.nan
        # 5) duplicate submissions
        dup = df.sample(k(0.015), random_state=seed)
        df = pd.concat([df, dup], ignore_index=True)

    df.insert(0, "respondent_id", np.arange(1, len(df) + 1))
    return df


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "data/synthetic_survey.csv"
    d = generate()
    d.to_csv(out, index=False)
    print(f"Wrote {len(d)} rows x {d.shape[1]} cols -> {out}")
