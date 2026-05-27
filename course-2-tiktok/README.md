# Course 2: Get Started with Python
## TikTok — Exploratory Data Analysis & Visualization

**Status:** ✅ Complete  
**Scenario:** TikTok Claims Classification  
**Deliverable type:** EDA Python notebook + Tableau dashboard + Executive Summary

---

## Overview

In Course 2, I performed Exploratory Data Analysis (EDA) on the TikTok dataset and created visualizations for both technical teammates (Python/matplotlib/seaborn) and non-technical stakeholders (Tableau dashboard).

This stage follows the **Analyze** and **Construct** stages of the PACE framework — taking the raw dataset from Course 1 and transforming it into structured insights through visualization.

---

## Stakeholder Requests

From **Orion Rainier (Data Scientist):**
- Python notebook with EDA, data cleaning, and visualizations
- At minimum: claim vs opinion count, boxplots of key variables, author ban status breakdown
- Tableau dashboard: scatter plot (views vs likes), claim/opinion count, author status
- Dashboard must be accessible to users with visual impairments

From **Willow Jaffey (Data Science Lead):**
- Executive summary of EDA findings

---

## Data Cleaning Decisions

| Issue | Finding | Decision |
|-------|---------|----------|
| Null values | 200 rows (1.03%) missing across 7 columns simultaneously | Dropped for EDA — systematic pattern, not random |
| Duplicates | 0 duplicate rows, 0 duplicate video IDs | No action needed |
| Outliers | Heavy right skew in all engagement metrics | Kept for EDA; log-transform recommended for modeling |

**Working dataset after cleaning: 19,182 rows × 12 columns**

---

## Key EDA Findings

### 1. Near-balanced target variable
```
claim      9,670  (50.4%)
opinion    9,512  (49.6%)
```
Favorable for unbiased ML classification — no resampling needed.

### 2. Engagement metrics strongly correlate with claim status

| Metric | Claim (median) | Opinion (median) | Difference |
|--------|---------------|-----------------|-----------|
| video_view_count | ~375,000 | ~8,000 | Claims ~47x higher |
| video_like_count | ~125,000 | ~2,000 | Claims ~62x higher |
| video_comment_count | ~850 | ~100 | Claims ~8x higher |

### 3. All engagement variables are heavily right-skewed
Most videos cluster at low counts; a small number of viral videos create extreme outliers at the high end.

### 4. Ban status correlates with claim status
Banned authors disproportionately post claim videos. This relationship between `author_ban_status` and `claim_status` is a potentially strong predictive signal for the classification model.

### 5. Video duration shows no meaningful difference
`video_duration_sec` ranges 5–60 seconds uniformly for both claims and opinions — not a useful classification feature.

---

## Outlier Strategy

| Phase | Approach | Rationale |
|-------|---------|-----------|
| EDA | Keep all outliers | Understand full distribution |
| Modeling | Log-transform engagement variables | Normalizes skewed distributions |
| Avoid | Dropping outlier rows | High-engagement videos may be disproportionately claims — removing would bias model |

---

## Visualizations Created

### Python (matplotlib/seaborn)
| Chart | Insight |
|-------|---------|
| Claim vs Opinion count bar chart | Near-balanced 50.4/49.6 split |
| Author ban status bar chart | 81.3% active, 11.2% under scrutiny, 8.5% banned |
| Boxplots by claim status (4 variables) | Claim videos dramatically outperform on engagement |
| Histograms (view, like, comment counts) | All heavily right-skewed |
| Claim by ban status grouped bar | Banned authors disproportionately post claims |

### Tableau (see dashboard link below)
| Chart | Type |
|-------|------|
| Views vs Likes scatter plot | Scatter — colored by claim_status |
| Claim vs Opinion count | Pie/bar chart |
| Author ban status breakdown | Bar chart |
| Key variable boxplots | Boxplot dashboard |

---

## Tableau Dashboard

🔗 **[Tableau Public link ](https://public.tableau.com/views/TiktokEndofCourse2Project/Story1?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)**

*Accessibility note: Dashboard uses both color and shape encoding on the scatter plot to ensure usability for viewers with visual impairments.*

---

## Deliverables

| File | Description |
|------|-------------|
| [TikTok_Course2_EDA.ipynb](TikTok_Course2_EDA.ipynb) | Complete EDA notebook with all Python visualizations |
| [PACE_Strategy_Document_Course2.md](PACE_Strategy_Document_Course2.md) | PACE document covering Plan, Analyze, Construct, Execute stages |
| [TikTok_Course2_Executive_Summary.pdf](TikTok_Course2_Executive_Summary.pdf) | One-page executive summary for stakeholders |
| [images/](images/) | All chart exports from the Python notebook |

---

## Python Skills Demonstrated

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Data cleaning
data_clean = data.dropna()
data.duplicated().sum()

# Distribution analysis
data_clean['claim_status'].value_counts(normalize=True)
data_clean['author_ban_status'].value_counts()

# Cross-tabulation
pd.crosstab(data_clean['author_ban_status'],
            data_clean['claim_status'], margins=True)

# Boxplots by category
sns.boxplot(data=data_clean, x='claim_status', y='video_view_count')

# Histograms
plt.hist(data_clean['video_view_count'], bins=50)

# Grouped bar charts
data_clean.groupby(['author_ban_status', 'claim_status']).size()
           .unstack().plot(kind='bar')
```

---

*Previous: [Course 1 — Data Inspection](../course-1-tiktok/) · Next: Course 3 — Translate Data into Insights*
