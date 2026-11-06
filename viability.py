"""Business-viability engine: turns model output + (editable) cost assumptions into a verdict.

All defaults below are PLACEHOLDER assumptions to be replaced with real quotes
(supplier price, ad CAC, salaries). They are deliberately not tuned to produce a good verdict.
"""
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd


@dataclass
class Assumptions:
    reachable_audience: int = 150_000      # people you can realistically reach with marketing in year 1
    intent_to_purchase: float = 0.25       # share of "interested" who actually buy (stated intent is inflated)
    spend_haircut: float = 0.70            # stated spend vs real spend
    planned_tshirt_price: float = 59.0     # AED, the price you plan for a basic tee
    gross_margin: float = 0.50             # after sustainable-fabric cost premium
    fulfilment_pct: float = 0.09           # delivery + packing, % of revenue
    payment_pct: float = 0.025             # payment gateway fees
    returns_loss_pct: float = 0.05         # margin lost to returns (online fashion returns are high)
    annual_retention: float = 0.40         # share of customers who buy again next year
    cac: float = 90.0                      # AED paid to acquire one customer (broad campaign)
    fixed_costs_year: float = 1_200_000.0  # team, platform, pop-ups, tools (AED / year)

    @property
    def contribution_margin(self):
        return self.gross_margin - self.fulfilment_pct - self.payment_pct - self.returns_loss_pct

    def as_dict(self):
        return asdict(self)


def unit_economics(R, a: Assumptions):
    d, reg, clu = R["data"], R["reg"], R["clu"]
    interested = d["interested"] == 1
    p_all = float(interested.mean())
    seg_p = clu["summary"]["actually_interested"]
    p_high = float(seg_p.iloc[0])
    pred = reg["pred_all"][interested]
    annual_spend = float(pred.median() * 2 * a.spend_haircut)
    cm = a.contribution_margin
    lifetime_years = min(1 / (1 - a.annual_retention), 5)
    contrib_per_year = annual_spend * cm
    ltv = contrib_per_year * lifetime_years

    def scenario(cac):
        customers = a.reachable_audience * p_all * a.intent_to_purchase
        revenue = customers * annual_spend
        contribution = revenue * cm
        marketing = customers * cac
        profit = contribution - marketing - a.fixed_costs_year
        margin_per_cust = contrib_per_year - cac
        be = a.fixed_costs_year / margin_per_cust if margin_per_cust > 0 else np.inf
        return {"customers": customers, "revenue": revenue, "contribution": contribution, "marketing": marketing,
                "fixed": a.fixed_costs_year, "profit_y1": profit, "breakeven_customers": be,
                "ltv_cac": ltv / cac if cac > 0 else np.inf, "cac": cac}

    broad = scenario(a.cac)
    cac_t = a.cac * p_all / max(p_high, 1e-9)       # pay only to reach people the model scores as high-probability
    targeted = scenario(cac_t)
    # the targeted scenario can only reach the high-probability pool, so cap the customers it can win
    pool = a.reachable_audience * float(clu["summary"]["share"].iloc[0])
    cap = pool * p_high * a.intent_to_purchase
    if targeted["customers"] > cap:
        scale = cap / targeted["customers"]
        targeted.update(customers=cap, revenue=targeted["revenue"] * scale, contribution=targeted["contribution"] * scale,
                        marketing=targeted["marketing"] * scale)
        targeted["profit_y1"] = targeted["contribution"] - targeted["marketing"] - a.fixed_costs_year
    return {"p_interested": p_all, "p_high_segment": p_high, "annual_spend": annual_spend, "cm": cm,
            "lifetime_years": lifetime_years, "ltv": ltv, "broad": broad, "targeted": targeted, "cac_targeted": cac_t}


def scorecard(R, a: Assumptions, ue=None):
    ue = ue or unit_economics(R, a)
    d, clf, asr = R["data"], R["clf"], R["asr"]
    inter = d[d["interested"] == 1]
    checks = []

    def add(name, value, thr, ok, why, fmt="{:.0%}"):
        checks.append({"Check": name, "Result": fmt.format(value) if isinstance(value, (int, float, np.floating)) else value,
                       "Pass mark": thr, "Pass": bool(ok), "Why it matters": why})

    p = ue["p_interested"]
    add("1. Demand: share clearly interested (intent 4-5)", p, ">= 30%", p >= 0.30, "Niche brands need a big enough keen minority")
    prem = float((d["q16_premium_pct"].fillna(0) >= 10).mean())
    add("2. Pays a sustainability premium (>=10%)", prem, ">= 35%", prem >= 0.35, "Eco fabrics cost more; customers must carry it")
    wtp_ok = float((inter["q15_wtp_tshirt_aed"].fillna(0) >= a.planned_tshirt_price).mean())
    add(f"3. Interested people accept AED {a.planned_tshirt_price:.0f} tee", wtp_ok, ">= 50%", wtp_ok >= 0.5, "Price must sit inside willingness-to-pay")
    auc = float(clf["metrics"].loc[clf["best"], "CV AUC (5-fold)"])
    add("4. Conversion model is reliable (AUC)", auc, ">= 0.75", auc >= 0.75, "Targeting only works if the model separates buyers", "{:.2f}")
    cov = float(asr["curve"].loc[asr["recommended_k"] - 1, "interest_covered"])
    add(f"5. Lean launch ({asr['recommended_k']} categories) covers demand", cov, ">= 75% in <= 6", cov >= 0.75 and asr["recommended_k"] <= 6, "Baby-step launch without missing 'C'")
    lc = ue["broad"]["ltv_cac"]
    add("6. LTV : CAC (broad campaign)", lc, ">= 3.0", lc >= 3, "Each customer must repay acquisition 3x", "{:.1f}")
    be, cu = ue["broad"]["breakeven_customers"], ue["broad"]["customers"]
    add("7. Year-1 customers exceed break-even", cu / be if np.isfinite(be) and be > 0 else 0.0, ">= 1.0x", np.isfinite(be) and cu >= be,
        "Fixed costs must be covered by customers you can win", "{:.2f}x")
    df = pd.DataFrame(checks)
    n = int(df["Pass"].sum())
    if n >= 6:
        verdict, tone = "VIABLE", "green"
    elif n >= 4:
        verdict, tone = "VIABLE WITH CONDITIONS", "orange"
    else:
        verdict, tone = "NOT YET VIABLE", "red"
    return df, n, verdict, tone
