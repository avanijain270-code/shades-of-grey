"""Single source of truth: survey question -> dataset column -> algorithm."""

CATEGORIES = {  # Q19 (multi-select) -> association rule mining items
    "q19_tshirts": "T-shirts & tanks",
    "q19_hoodies": "Hoodies & sweatshirts",
    "q19_trousers": "Trousers & jeans",
    "q19_outerwear": "Jackets & outerwear",
    "q19_formal": "Formal / office wear",
    "q19_knitwear": "Knitwear",
    "q19_activewear": "Activewear",
    "q19_accessories": "Accessories (scarves, caps, bags)",
}
FEATURES = {  # Q20 (multi-select) -> association rule mining items
    "q20_subscription_box": "Curated subscription box",
    "q20_limited_editions": "Limited-edition drops",
    "q20_eco_certified": "Eco-certified fabrics label",
    "q20_styling_guide": "Mix-and-match styling guide",
    "q20_loyalty": "Loyalty rewards",
    "q20_free_returns": "Free returns",
    "q20_custom_fit": "Made-to-measure fit",
}
ITEMS = {**CATEGORIES, **FEATURES}

LIKERT = ["q10_grey_pref", "q11_sustain_importance", "q12_brand_loyalty",
          "q13_concept_appeal", "q14_price_fairness", "q17_switch_likelihood",
          "q18_trust_claims", "q21_purchase_intent", "q29_influencer"]

NUMERIC_OUTLIER_COLS = ["q05_income_aed", "q07_clothing_spend_aed", "q09_avg_item_price_aed",
                        "q15_wtp_tshirt_aed", "q25_spend_6m_aed", "q27_sub_max_aed",
                        "q06_orders_per_month", "q26_orders_per_year"]

TARGET_CLASS = "interested"          # derived from Q21 (>=4 on a 1-5 scale)
TARGET_REG = "q25_spend_6m_aed"      # Q25

CAT_COLS = ["q01_age_group", "q02_gender", "q03_residence", "q04_occupation",
            "q08_primary_brand", "q28_discovery", "q22_subscribe", "q23_popup"]

# Outcome questions would leak the answer, so they never predict intent
LEAKY_FOR_CLASSIFICATION = ["q21_purchase_intent", "q22_subscribe", "q24_nps", "q25_spend_6m_aed",
                            "q26_orders_per_year", "q27_sub_max_aed", "interested", "q30_open_text"]
LEAKY_FOR_REGRESSION = ["q26_orders_per_year", "q27_sub_max_aed", "q24_nps", "interested", "q30_open_text"]

# (Q#, section, question, answer type, column(s), algorithm(s))
SURVEY = [
 (1,"A. Profile","Which age group are you in?","Single choice: 16-24 / 25-34 / 35-50","q01_age_group","Clustering, Classification"),
 (2,"A. Profile","What is your gender?","Single choice: Female / Male / Prefer not to say","q02_gender","Clustering, Classification"),
 (3,"A. Profile","Where do you live?","Single choice: Metro core / Suburban / Other","q03_residence","Clustering, Classification"),
 (4,"A. Profile","What is your occupation?","Single choice: Student / Employed / Self-employed / Other","q04_occupation","Clustering, Classification"),
 (5,"A. Profile","Approximate monthly income / allowance (AED)?","Number","q05_income_aed","Clustering, Regression"),
 (6,"A. Profile","How many online orders (any category) do you place per month?","Number","q06_orders_per_month","Clustering, Classification"),
 (7,"B. Current behaviour","How much do you spend on clothing per month (AED)?","Number","q07_clothing_spend_aed","Regression (strongest predictor of future spend)"),
 (8,"B. Current behaviour","Which brand do you buy clothes from most often?","Single choice: H&M / Zara / Uniqlo / Shein / Local-other","q08_primary_brand","Classification, Clustering"),
 (9,"B. Current behaviour","Average price you pay per clothing item (AED)?","Number","q09_avg_item_price_aed","Regression, Pricing"),
 (10,"B. Current behaviour","I prefer neutral / grey-scale colours in my wardrobe.","Likert 1-5","q10_grey_pref","Classification, Clustering"),
 (11,"B. Current behaviour","Sustainability of fabrics matters when I buy clothes.","Likert 1-5","q11_sustain_importance","Classification, Clustering"),
 (12,"B. Current behaviour","I am loyal to the brands I currently buy from.","Likert 1-5","q12_brand_loyalty","Classification (switching barrier)"),
 (13,"C. Concept test","[Show concept card] How appealing is a brand that sells ONLY grey-scale clothing?","Likert 1-5","q13_concept_appeal","Classification, Clustering"),
 (14,"C. Concept test","Prices similar to / slightly below H&M for premium sustainable quality are fair.","Likert 1-5","q14_price_fairness","Classification, Regression"),
 (15,"C. Concept test","Maximum you would pay for one grey-scale T-shirt (AED)?","Number","q15_wtp_tshirt_aed","Regression, Pricing, Clustering"),
 (16,"C. Concept test","What extra % would you pay for sustainable fabrics? (0/5/10/15/20/25)","Single choice","q16_premium_pct","Pricing, Classification"),
 (17,"C. Concept test","How likely are you to switch part of your spend away from your current brand?","Likert 1-5","q17_switch_likelihood","Classification"),
 (18,"C. Concept test","I trust a brand's sustainability claims if certified.","Likert 1-5","q18_trust_claims","Classification"),
 (19,"D. Product interest","Which categories would you buy? (select all)","Multi-select x8","q19_*","Association rule mining"),
 (20,"D. Product interest","Which features/services interest you? (select all)","Multi-select x7","q20_*","Association rule mining"),
 (21,"E. Conversion","How likely are you to buy from us in the next 3 months?","Likert 1-5 (4-5 = interested) -> TARGET","q21_purchase_intent","Classification (target), Clustering (ranking)"),
 (22,"E. Conversion","Would you subscribe to a monthly curated grey-scale box?","Single choice: Yes / Maybe / No","q22_subscribe","Association, Revenue stream test"),
 (23,"E. Conversion","Would you attend / buy at a pop-up store?","Single choice: Yes / No","q23_popup","Classification, Channel strategy"),
 (24,"E. Conversion","How likely are you to recommend us to a friend (0-10)?","Number 0-10","q24_nps","Validation / NPS"),
 (25,"F. Spend","How much would you spend with us in the first 6 months (AED)?","Number -> TARGET","q25_spend_6m_aed","Regression (target)"),
 (26,"F. Spend","How many orders per year would you place?","Number","q26_orders_per_year","Unit economics"),
 (27,"F. Spend","Maximum monthly price for a subscription box (AED)?","Number","q27_sub_max_aed","Pricing"),
 (28,"G. Marketing","Where would you most likely discover us?","Single choice: Instagram / TikTok / Facebook / Friends / Pop-up / Search","q28_discovery","Channel strategy, Classification"),
 (29,"G. Marketing","Influencers affect what I buy.","Likert 1-5","q29_influencer","Classification, Clustering"),
 (30,"G. Marketing","In your own words: what would stop you from buying from us?","Open text","q30_open_text","Text analytics (session 8 homework)"),
]
