# Shades of Grey - Consumer Survey (30 questions)

Every question feeds at least one algorithm. Q21 and Q25 are the two model targets.


## A. Profile

**Q1. Which age group are you in?**  
*Single choice: 16-24 / 25-34 / 35-50* - `q01_age_group` - used by: Clustering, Classification

**Q2. What is your gender?**  
*Single choice: Female / Male / Prefer not to say* - `q02_gender` - used by: Clustering, Classification

**Q3. Where do you live?**  
*Single choice: Metro core / Suburban / Other* - `q03_residence` - used by: Clustering, Classification

**Q4. What is your occupation?**  
*Single choice: Student / Employed / Self-employed / Other* - `q04_occupation` - used by: Clustering, Classification

**Q5. Approximate monthly income / allowance (AED)?**  
*Number* - `q05_income_aed` - used by: Clustering, Regression

**Q6. How many online orders (any category) do you place per month?**  
*Number* - `q06_orders_per_month` - used by: Clustering, Classification


## B. Current behaviour

**Q7. How much do you spend on clothing per month (AED)?**  
*Number* - `q07_clothing_spend_aed` - used by: Regression (strongest predictor of future spend)

**Q8. Which brand do you buy clothes from most often?**  
*Single choice: H&M / Zara / Uniqlo / Shein / Local-other* - `q08_primary_brand` - used by: Classification, Clustering

**Q9. Average price you pay per clothing item (AED)?**  
*Number* - `q09_avg_item_price_aed` - used by: Regression, Pricing

**Q10. I prefer neutral / grey-scale colours in my wardrobe.**  
*Likert 1-5* - `q10_grey_pref` - used by: Classification, Clustering

**Q11. Sustainability of fabrics matters when I buy clothes.**  
*Likert 1-5* - `q11_sustain_importance` - used by: Classification, Clustering

**Q12. I am loyal to the brands I currently buy from.**  
*Likert 1-5* - `q12_brand_loyalty` - used by: Classification (switching barrier)


## C. Concept test

**Q13. [Show concept card] How appealing is a brand that sells ONLY grey-scale clothing?**  
*Likert 1-5* - `q13_concept_appeal` - used by: Classification, Clustering

**Q14. Prices similar to / slightly below H&M for premium sustainable quality are fair.**  
*Likert 1-5* - `q14_price_fairness` - used by: Classification, Regression

**Q15. Maximum you would pay for one grey-scale T-shirt (AED)?**  
*Number* - `q15_wtp_tshirt_aed` - used by: Regression, Pricing, Clustering

**Q16. What extra % would you pay for sustainable fabrics? (0/5/10/15/20/25)**  
*Single choice* - `q16_premium_pct` - used by: Pricing, Classification

**Q17. How likely are you to switch part of your spend away from your current brand?**  
*Likert 1-5* - `q17_switch_likelihood` - used by: Classification

**Q18. I trust a brand's sustainability claims if certified.**  
*Likert 1-5* - `q18_trust_claims` - used by: Classification


## D. Product interest

**Q19. Which categories would you buy? (select all)**  
*Multi-select x8* - `q19_*` - used by: Association rule mining

**Q20. Which features/services interest you? (select all)**  
*Multi-select x7* - `q20_*` - used by: Association rule mining


## E. Conversion

**Q21. How likely are you to buy from us in the next 3 months?**  
*Likert 1-5 (4-5 = interested) -> TARGET* - `q21_purchase_intent` - used by: Classification (target), Clustering (ranking)

**Q22. Would you subscribe to a monthly curated grey-scale box?**  
*Single choice: Yes / Maybe / No* - `q22_subscribe` - used by: Association, Revenue stream test

**Q23. Would you attend / buy at a pop-up store?**  
*Single choice: Yes / No* - `q23_popup` - used by: Classification, Channel strategy

**Q24. How likely are you to recommend us to a friend (0-10)?**  
*Number 0-10* - `q24_nps` - used by: Validation / NPS


## F. Spend

**Q25. How much would you spend with us in the first 6 months (AED)?**  
*Number -> TARGET* - `q25_spend_6m_aed` - used by: Regression (target)

**Q26. How many orders per year would you place?**  
*Number* - `q26_orders_per_year` - used by: Unit economics

**Q27. Maximum monthly price for a subscription box (AED)?**  
*Number* - `q27_sub_max_aed` - used by: Pricing


## G. Marketing

**Q28. Where would you most likely discover us?**  
*Single choice: Instagram / TikTok / Facebook / Friends / Pop-up / Search* - `q28_discovery` - used by: Channel strategy, Classification

**Q29. Influencers affect what I buy.**  
*Likert 1-5* - `q29_influencer` - used by: Classification, Clustering

**Q30. In your own words: what would stop you from buying from us?**  
*Open text* - `q30_open_text` - used by: Text analytics (session 8 homework)
