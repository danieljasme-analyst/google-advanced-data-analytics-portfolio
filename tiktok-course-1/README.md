# Course 1: Foundations of Data Science
## TikTok — Initial Data Inspection and Organization

**Status:** ✅ Complete  
**Scenario:** TikTok Claims Classification  
**Deliverable type:** Python notebook + PACE Strategy Document + Executive Summary

---

## Overview

In this course, I completed the initial data inspection phase of the TikTok claims classification project. Before any modeling or analysis could begin, the dataset needed to be loaded, inspected, and understood — this is the foundation every data project is built on.

This phase follows the **Plan** and **Analyze** stages of the PACE framework. The Construct and Execute stages (model building and communication) will be developed in later courses.

---

## Business Context

TikTok's Trust & Safety team needs a machine learning model that can automatically classify whether a video contains a **claim** (unverified information presented as fact) or an **opinion** (personal belief or thought). Manual review at the platform's scale is not feasible.

My task in Course 1: inspect the provided dataset, understand its structure and quality, engineer initial features, and communicate findings to the data team through an executive summary.

---

## Stakeholders

| Name | Role |
|------|------|
| Rosie Mae Bradshaw | Data Science Manager — project sponsor |
| Orion Rainier | Data Scientist — technical lead, requested the data inspection |
| Willow Jaffey | Data Science Lead — strategic oversight |

---

## Key Findings

| Finding | Detail |
|---------|--------|
| Dataset size | 19,383 rows × 12 columns |
| Claim/opinion split | ~49.6% claims / ~50.4% opinions — near-balanced ✅ |
| Null values | 298 rows (~1.54%) missing all 5 engagement columns |
| Video duration range | 5–60 seconds — no anomalies |
| View count range | ~20 to ~999,817 — heavy right skew |
| Claim videos engagement | likes_per_view ≈ 0.33 vs opinion videos ≈ 0.22 |
| Banned authors | ~2x more raw views than active authors |

---

## New Features Engineered

Three normalized engagement rate columns were created to enable fair comparison across videos with different view counts:

```python
data['likes_per_view']    = data['video_like_count']    / data['video_view_count']
data['comments_per_view'] = data['video_comment_count'] / data['video_view_count']
data['shares_per_view']   = data['video_share_count']   / data['video_view_count']
```

**Key insight from these features:** Claim videos consistently have higher engagement *rates* than opinion videos (~50% higher likes_per_view), and this pattern holds across all three author ban status categories. Claim status is a stronger predictor of engagement rate than ban status.

---

## Deliverables

| File | Description |
|------|-------------|
| [TikTok_Course1_Project.ipynb](TikTok_Course1_Project.ipynb) | Complete Python notebook — data loading, inspection, variable analysis, feature engineering |
| [PACE_Strategy_Document.md](PACE_Strategy_Document.md) | PACE framework document covering Plan, Analyze, and Execute stages |
| [TikTok_Course1_Executive_Summary.pdf](TikTok_Course1_Executive_Summary.pdf) | One-page professional summary for the TikTok data team |

---

## Python Skills Demonstrated

```python
# Core libraries
import pandas as pd
import numpy as np

# Data inspection
data.head(10)
data.info()
data.describe()

# Variable analysis
data['claim_status'].value_counts(normalize=True)
data['author_ban_status'].value_counts()

# Grouped analysis
data.groupby(['author_ban_status']).agg({'video_view_count': ['count', 'mean', 'median']})
data.groupby(['claim_status', 'author_ban_status']).agg(
    {'likes_per_view': ['count', 'mean', 'median'],
     'comments_per_view': ['count', 'mean', 'median'],
     'shares_per_view': ['count', 'mean', 'median']})

# Feature engineering
data['likes_per_view']    = data['video_like_count']    / data['video_view_count']
data['comments_per_view'] = data['video_comment_count'] / data['video_view_count']
data['shares_per_view']   = data['video_share_count']   / data['video_view_count']
```

---

## Recommended Next Steps

1. **Handle null values** — 298 rows missing all engagement data; investigate before modeling
2. **Full EDA with visualizations** — distribution plots, correlation analysis
3. **NLP exploration** — analyze `video_transcription_text` for classification signals
4. **Investigate ban status ↔ claim status relationship** — these appear correlated
5. **Begin feature engineering pipeline** for hypothesis testing in Course 4

---

*Next: Course 2 — Python EDA and data visualization*
