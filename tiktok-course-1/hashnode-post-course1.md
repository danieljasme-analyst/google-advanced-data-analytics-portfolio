# Getting Started with TikTok Data: PACE Framework and Initial Inspection (Google Advanced DA Certificate — Course 1)

*This is the first post in my Google Advanced Data Analytics Certificate series, documenting the TikTok claims classification project. Each post covers one course's end-of-course project.*

---

## The Scenario

I recently started the Google Advanced Data Analytics Certificate on Coursera. Like the Google BI Certificate I completed previously, this program uses a consistent workplace scenario across all courses — building a complete project from start to finish rather than isolated exercises.

The scenario: I've just joined TikTok's data analytics team. The Trust & Safety team needs a machine learning model to classify whether statements made in videos are **claims** (unverified information presented as fact) or **opinions** (personal beliefs). My job is to build that model — starting from scratch with raw data.

Course 1 is the foundation stage. Before any analysis, visualization, or modeling, I need to understand what data I'm working with.

---

## The PACE Framework

Every project in this certificate follows the **PACE** framework — a structured approach to data work:

| Stage | What it means |
|-------|--------------|
| **P**lan | Understand the situation, define scope, identify stakeholders |
| **A**nalyze | Inspect, clean, and explore the data |
| **C**onstruct | Build models or visualizations |
| **E**xecute | Share results and communicate findings |

Course 1 sits entirely in the **Plan** and early **Analyze** stages. The Construct and Execute stages come in later courses once we have a proper understanding of the data.

---

## The Dataset

The dataset is `tiktok_dataset.csv` — synthetic data created in partnership with TikTok for educational purposes.

- **19,383 rows** — one per TikTok video flagged as containing a claim or opinion
- **12 columns** — covering claim status, video metadata, author information, and engagement metrics
- **Target variable:** `claim_status` — either "claim" or "opinion"

The five columns that matter most for the future model:

```python
data.info()
# claim_status              object   ← target variable
# video_transcription_text  object   ← key for NLP
# author_ban_status         object   ← categorical signal
# video_view_count          float64  ← engagement
# video_like_count          float64  ← engagement
```

---

## Three Things I Found Immediately

**1. The target is nearly balanced**

```python
data['claim_status'].value_counts(normalize=True)
# claim      0.496
# opinion    0.504
```

50/50 split. This matters a lot for machine learning — an imbalanced dataset (say 90% claims, 10% opinions) would make it easy for a lazy model to achieve high accuracy just by always predicting the majority class. A balanced dataset means the model actually has to learn the difference.

**2. There are 298 rows with missing engagement data**

```python
data.isnull().sum()
# video_view_count      298
# video_like_count      298
# video_share_count     298
# video_download_count  298
# video_comment_count   298
```

All five engagement columns are missing for the same 298 rows simultaneously. That's not random — it's systematic. These videos likely had a data collection failure. At 1.54% of the dataset it's small enough to drop for initial analysis, but worth investigating before the modeling phase.

**3. Engagement metrics are highly skewed**

```python
data['video_view_count'].describe()
# mean      253,763
# 50%        75,000   ← median
# max       999,817
```

The mean is more than 3x the median. A small number of viral videos are pulling the average way up. This kind of right skew is typical for social media data and means we'll need to log-transform these variables before feeding them into a model.

---

## New Features Created

The team asked me to create meaningful variables by combining existing ones. I added three normalized engagement rate columns:

```python
data['likes_per_view']    = data['video_like_count']    / data['video_view_count']
data['comments_per_view'] = data['video_comment_count'] / data['video_view_count']
data['shares_per_view']   = data['video_share_count']   / data['video_view_count']
```

The key finding: **claim videos have ~50% higher likes-per-view than opinion videos** (0.33 vs 0.22), and this pattern holds regardless of the author's ban status. Claim status genuinely drives engagement behavior — not just the author's account standing.

---

## What's Next

Course 2 is where the data starts talking. Full EDA with matplotlib and seaborn visualizations, proper cleaning decisions, and a Tableau dashboard for non-technical stakeholders.

---

## Portfolio Links

- 📁 **GitHub:** [github.com/danieljasme-analyst/google-advanced-data-analytics-portfolio](https://github.com/danieljasme-analyst/google-advanced-data-analytics-portfolio)

---

*Tags: python, data-analytics, google-certificate, tiktok, pandas, jupyter, pace-framework, portfolio*
