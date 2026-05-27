# Google-Advanced-Data-Analytics-portfolio
 End-of-course portfolio projects for the Google Advanced Data Analytics Certificate — TikTok claims classification case study using Pyt# Google Advanced Data Analytics Certificate — Portfolio

**Candidate:** Ahmad Daniel  
**Program:** [Google Advanced Data Analytics Certificate](https://grow.google/certificates/advanced-data-analytics/) on Coursera  
**Blog:** [danieljasme.hashnode.dev](https://danieljasme.hashnode.dev)  
**Status:** 🔄 In progress

---

## About This Portfolio

This repository documents my end-of-course projects from the **Google Advanced Data Analytics Certificate**. Each course builds on the last — simulating the full lifecycle of a real data project from initial data inspection through to a deployed machine learning model.

All projects follow the **TikTok** scenario: a fictional case study where I am a new member of TikTok's data analytics team, tasked with building a machine learning model to classify whether statements made in videos are claims or opinions.

---

## Project Scenario: TikTok Claims Classification

> TikTok is the leading destination for short-form mobile video. The platform's Trust & Safety team needs a machine learning model to classify whether statements made in videos are **claims** (unverified information) or **opinions** (personal beliefs). This will help streamline the content moderation process at scale.

**My role:** Data professional responsible for the full project lifecycle — from data inspection and EDA through statistical analysis, regression modeling, and final machine learning model deployment.

**Project goal:**  
*Build a machine learning model that can accurately classify TikTok videos as containing a claim or an opinion.*

---

## Certificate Structure & Deliverables

| Course | Title | Status | Key Deliverables |
|--------|-------|--------|-----------------|
| Course 1 | Foundations of Data Science | ✅ Complete | PACE document, Python notebook (data inspection), Executive Summary |
| Course 2 | Get Started with Python | ✅ Complete  | EDA notebook with visualizations |
| Course 3 | Go Beyond the Numbers: Translate Data into Insights | ⏳ Upcoming | Full EDA, data visualizations |
| Course 4 | The Power of Statistics | ⏳ Upcoming | Hypothesis testing, statistical analysis |
| Course 5 | Regression Analysis: Simplify Complex Data Relationships | ⏳ Upcoming | Regression model |
| Course 6 | The Nuts and Bolts of Machine Learning | ⏳ Upcoming | ML classification model |
| Capstone | Google Advanced Data Analytics Capstone | ⏳ Upcoming | Full end-to-end portfolio project |

---

## The Dataset

`tiktok_dataset.csv` — Synthetic data created in partnership with TikTok for educational purposes.

| Property | Detail |
|----------|--------|
| Rows | 19,383 (one per TikTok video) |
| Columns | 12 |
| Target variable | `claim_status` — "claim" or "opinion" |
| Split | ~49.6% claims / ~50.4% opinions |
| Null values | 298 rows missing engagement data |

**Column reference:**

| Column | Type | Description |
|--------|------|-------------|
| `#` | int | Row index |
| `claim_status` | object | Target: "claim" or "opinion" |
| `video_id` | int | Unique video identifier |
| `video_duration_sec` | int | Video length in seconds (5–60) |
| `video_transcription_text` | object | Spoken content — key for NLP |
| `verified_status` | object | "verified" or "not verified" |
| `author_ban_status` | object | "active", "under review", or "banned" |
| `video_view_count` | float | Total views |
| `video_like_count` | float | Total likes |
| `video_share_count` | float | Total shares |
| `video_download_count` | float | Total downloads |
| `video_comment_count` | float | Total comments |

---

## PACE Framework

All projects in this certificate follow the **PACE** problem-solving framework:

| Stage | Description |
|-------|-------------|
| **P**lan | Define scope, understand the situation, identify stakeholders |
| **A**nalyze | Collect, clean, and explore data |
| **C**onstruct | Build models, test hypotheses |
| **E**xecute | Share insights, communicate results, recommend next steps |

---

## Skills Demonstrated

- Python (pandas, numpy, matplotlib, seaborn, scikit-learn)
- Exploratory Data Analysis (EDA)
- Data cleaning and feature engineering
- Statistical hypothesis testing
- Regression analysis (linear and logistic)
- Machine learning model development and evaluation
- Professional communication and stakeholder reporting
- PACE framework for structured data project management

---

## Tools Used

| Tool | Purpose |
|------|---------|
| Python 3 | Primary programming language |
| Jupyter Notebook | Interactive coding environment |
| pandas / numpy | Data manipulation |
| matplotlib / seaborn | Data visualization |
| scikit-learn | Machine learning |
| GitHub | Portfolio hosting and version control |
| Hashnode | Project writeups and blog posts |

---

## Related Portfolio

**Google Business Intelligence Certificate:**  
[github.com/danieljasme-analyst/google-bi-portfolio](https://github.com/danieljasme-analyst/google-bi-portfolio) — Tableau dashboards, BigQuery SQL pipelines, and BI planning documents using the Google Fiber call center case study.

---

## Contact

**Blog:** [danieljasme.hashnode.dev](https://danieljasme.hashnode.dev)  
**LinkedIn:** *(Coming soon — updating after completing all certificates)*

---

*The TikTok scenario is a fictional case study provided by the Google Advanced Data Analytics Certificate program for educational purposes. All data is synthetic.*
hon, EDA, statistics, regression, and machine learning.
