# Exploring TikTok Data with Python: My EDA Approach (Google Advanced DA Certificate — Course 2)

*This is the second post in my Google Advanced Data Analytics Certificate series. [Part 1 covers the initial data inspection (Course 1)](https://danieljasme.hashnode.dev). This post covers Exploratory Data Analysis (EDA) with Python and Tableau.*

---

## What Changed Between Course 1 and Course 2

In Course 1, I loaded the TikTok dataset, inspected its structure, and created three basic derived features. Useful groundwork, but not yet insightful.

Course 2 is where the data starts telling a story.

EDA — Exploratory Data Analysis — is the process of understanding your data before you model it. It's part detective work, part quality control, part storytelling. The goal isn't to produce a final answer, it's to ask the right questions so your eventual model is built on solid understanding rather than assumptions.

---

## The Business Context

TikTok's Trust & Safety team needs a machine learning model to classify whether videos contain **claims** (unverified information presented as fact) or **opinions** (personal beliefs). My role is to prepare the data and surface the patterns that will inform that model.

At this stage, the data team — specifically Orion Rainier (Data Scientist) and Willow Jaffey (Data Science Lead) — needed:

- A Python notebook showing EDA, data cleaning, and visualizations
- A Tableau dashboard for non-technical stakeholders
- An executive summary of findings

---

## Step 1: Data Cleaning Decisions

Before visualizing anything, I had to handle the data quality issues flagged in Course 1.

**Null values:** 200 rows (1.03% of the dataset) had missing values across 7 columns simultaneously — `claim_status`, `video_transcription_text`, and all 5 engagement columns at once. This pattern is systematic, not random. When all engagement columns are missing together, it suggests those videos had a data collection failure, not individual field gaps.

Decision: **Drop these rows for EDA.** At 1.03% of the data, the loss is negligible. For the eventual modeling phase, I flagged that we need to investigate whether these nulls are associated with a particular claim type before making a permanent decision.

**Duplicates:** Zero duplicate rows, zero duplicate video IDs. Clean.

**Working dataset: 19,182 rows × 12 columns.**

---

## Step 2: Understanding the Target Variable

The first thing to check for any classification project is whether your target variable is balanced.

```python
data_clean['claim_status'].value_counts(normalize=True)
# claim      0.504
# opinion    0.496
```

Claims: 50.4%. Opinions: 49.6%. Nearly perfect balance. This is good news — it means we won't need to artificially resample the data to prevent the model from being biased toward the majority class.

---

## Step 3: The Distributions Tell a Story

I created histograms for the three key engagement variables: view count, like count, and comment count.

All three show the same pattern: extreme right skew. The vast majority of videos cluster at low values, with a long tail of viral videos pulling the mean far above the median.

For example:
- **Video view count:** median ~75,000 but mean ~253,000 — the mean is 3.4x the median
- **Video comment count:** median ~90 but mean ~342 — the mean is nearly 4x the median

This kind of skew is completely normal for social media data. A small number of videos go viral and distort the average. For modeling purposes, this means we'll likely need to **log-transform these variables** to normalize the distributions before feeding them into a regression or tree-based model.

---

## Step 4: The Key Finding — Claim Status Drives Engagement

This is where the EDA became genuinely interesting.

When I created boxplots of engagement metrics broken down by claim status, the difference was stark:

```python
sns.boxplot(data=data_clean, x='claim_status', y='video_view_count')
```

Claim videos have dramatically higher view counts, like counts, and comment counts than opinion videos. The median view count for claim videos is roughly **47x higher** than for opinion videos.

Two possible explanations:
1. Claims are inherently more engaging (controversy drives clicks)
2. Claims are more likely to be shared and spread virally

Either way, this is a strong signal for the classification model. Engagement metrics will likely be powerful predictive features.

---

## Step 5: Ban Status and Claim Status Are Related

The cross-tabulation of author ban status against claim status revealed another pattern:

```python
pd.crosstab(data_clean['author_ban_status'],
            data_clean['claim_status'], margins=True)
```

Among banned authors, claims significantly outnumber opinions. Among active authors, the split is much more balanced. This suggests that claim-making behavior is associated with the kind of conduct that gets authors banned or placed under scrutiny.

This is a useful signal, but it also creates a modeling risk — if we rely too heavily on `author_ban_status` as a feature, the model might classify any video by a banned author as a claim, regardless of content. That's a proxy variable, not a causal one.

---

## Step 6: Tableau Dashboard for Non-Technical Stakeholders

One specific request from the team was a **Tableau dashboard** — both because it's more interactive than static Python charts, and because it's easier for non-technical stakeholders like the operations and finance leads to explore on their own.

The dashboard I built includes:
- **Scatter plot:** views vs likes, colored by claim status — shows the engagement gap visually
- **Pie chart:** claim vs opinion count — confirms the balanced split
- **Bar chart:** author ban status breakdown

For accessibility (one team member has visual impairments), I used both **color and shape** encoding on the scatter plot — circles for claims, squares for opinions — so the chart is readable even without color distinction.

---

## The Outlier Question

The biggest methodological decision in this EDA was what to do about outliers.

The temptation is always to remove them — they make your distributions messy and your statistics harder to interpret. But in this case, **removing high-engagement outliers would introduce bias**.

Why? Because high-engagement videos are disproportionately claims. If we cap or remove them, we're systematically removing the most "claim-like" data points from our training set — which would make the eventual model worse at classifying claims.

My recommendation to the team:
- **EDA phase:** Keep all outliers to understand the full distribution
- **Modeling phase:** Log-transform engagement variables to normalize distributions without discarding data
- **Never:** Drop rows just because they have high engagement values

---

## Key Takeaways from Course 2 EDA

1. **The dataset is clean and balanced** — 1.03% null rows dropped, zero duplicates, 50.4/49.6 claim/opinion split
2. **Engagement metrics are the strongest signal** — claim videos get dramatically more views, likes, and comments
3. **Author ban status is correlated with claim status** — useful feature but needs careful handling to avoid proxy bias
4. **All engagement variables are right-skewed** — log-transform before modeling
5. **Video duration is not a useful feature** — no meaningful difference between claims and opinions

---

## What's Next

Course 3 moves into deeper data translation — working with more sophisticated visualization techniques and beginning to prepare data for statistical analysis. The patterns identified in this EDA will guide the hypothesis testing in Course 4.

---

## Portfolio Links

- 📁 **GitHub:** [github.com/danieljasme-analyst/google-advanced-data-analytics-portfolio](https://github.com/danieljasme-analyst/google-advanced-data-analytics-portfolio)
- 📊 **Tableau Dashboard:** [Add Tableau Public URL after publishing]
- 📝 **Course 1 post:** [Initial data inspection](https://danieljasme.hashnode.dev)

---

*Tags: python, exploratory-data-analysis, data-analytics, google-certificate, tiktok, pandas, seaborn, tableau, portfolio*
