# PACE Strategy Document — TikTok Course 1
## Foundations of Data Science

**Data Professional:** Ahmad Daniel  
**Project:** TikTok Claims Classification  
**Course:** 1 — Foundations of Data Science  
**Date:** May 2026

---

## What is PACE?

PACE is a problem-solving framework used by data professionals to structure their workflow:

| Stage | Description |
|-------|-------------|
| **P**lan | Define scope, understand the situation, identify stakeholders |
| **A**nalyze | Collect, prepare, and explore data |
| **C**onstruct | Build models, test hypotheses |
| **E**xecute | Share insights, communicate results, recommend next steps |

---

## PACE: Plan

### Understand the Situation

**What are the goals of this project?**  
The TikTok data team is developing a machine learning model to classify whether statements made in videos are claims or opinions. The immediate goal (Course 1) is to inspect, organize, and understand the data before any modeling begins.

**Who are the stakeholders?**

| Stakeholder | Role | Interest |
|-------------|------|---------|
| Rosie Mae Bradshaw | Data Science Manager | Project oversight; wants progress summary |
| Orion Rainier | Data Scientist | Technical lead; needs data inspected and organized |
| Willow Jaffey | Data Science Lead | Strategic direction of the claims classification project |
| TikTok Trust & Safety team | Internal client | Ultimately benefits from accurate claim classification |

**What are the questions we need to answer?**
- What does the dataset look like — rows, columns, data types?
- Are there any null values or data quality issues?
- What is the distribution of claim vs opinion videos?
- What variables might be most predictive for classification?
- What are the min/max ranges of key variables?

**What data do we have access to?**  
`tiktok_dataset.csv` — 19,383 rows × 12 columns of synthetic TikTok video data including claim status, video metadata, author status, and engagement metrics.

---

## PACE: Analyze

### Data Inspection Notes

**Dataset dimensions:** 19,383 rows × 12 columns

**Column data types:**

| Column | Type | Notes |
|--------|------|-------|
| `#` | int | Row index — not a feature |
| `claim_status` | object | Target variable: "claim" or "opinion" |
| `video_id` | int | Identifier — not a feature |
| `video_duration_sec` | int | Duration 5–60 seconds |
| `video_transcription_text` | object | Text data — key for NLP features |
| `verified_status` | object | "verified" or "not verified" |
| `author_ban_status` | object | "active", "under review", or "banned" |
| `video_view_count` | float | Has 298 null values |
| `video_like_count` | float | Has 298 null values |
| `video_share_count` | float | Has 298 null values |
| `video_download_count` | float | Has 298 null values |
| `video_comment_count` | float | Has 298 null values |

**Null value analysis:**  
298 rows (1.54%) have missing values across all 5 engagement columns simultaneously. This pattern suggests a systematic data collection issue rather than random missingness — these records may represent videos where engagement tracking failed.

**Key variable ranges:**
- `video_duration_sec`: min = 5 sec, max = 60 sec
- `video_view_count`: min ≈ 20, max ≈ 999,817

**Claim status distribution:**
- Claims: ~49.6% (≈ 9,608 videos)
- Opinions: ~50.4% (≈ 9,750 videos)
- Near-balanced — favorable for classification modeling

**Engagement by author ban status:**  
Banned authors and those under review receive significantly higher raw view counts, likes, and shares than active authors. However, when normalized per view (engagement rate), claim status becomes the stronger predictor.

**New derived variables created:**
- `likes_per_view` = video_like_count / video_view_count
- `comments_per_view` = video_comment_count / video_view_count
- `shares_per_view` = video_share_count / video_view_count

**Key pattern from derived variables:**  
Claim videos have consistently higher engagement rates than opinion videos (likes_per_view: ~0.33 vs ~0.22) — regardless of author ban status. This suggests claim status genuinely drives viewer engagement behavior.

---

## PACE: Construct

*Note: The Construct stage (model building) does not apply to Course 1. This stage will be addressed in later courses of the certificate when hypothesis testing, regression analysis, and machine learning model development begin.*

---

## PACE: Execute

### Summary for Stakeholders

**What percentage of the data is claims vs opinions?**  
Approximately 49.6% claims and 50.4% opinions — a near-balanced target variable, which is favorable for building an unbiased classification model.

**What factors correlate with a video's claim status?**  
- Author ban status: banned authors and those under review are disproportionately associated with claim videos
- Engagement rates: claim videos have higher likes_per_view, shares_per_view, and comments_per_view than opinion videos across all author ban categories
- Video transcription text: the text content will likely be the most direct indicator — NLP analysis should be prioritized

**What factors correlate with a video's engagement level?**  
- Author ban status is the strongest predictor of *raw* engagement volume (views, likes, shares)
- Claim status is the stronger predictor of *engagement rate* (likes per view, shares per view)
- These two variables appear correlated — banned authors post more claims — but provide distinct signals

### Recommended Next Steps

1. **Handle nulls**: Investigate the 298 null rows — confirm whether they are random or systematic, then decide to drop or impute
2. **Full EDA**: Visualize distributions and relationships between all key variables
3. **NLP exploration**: Begin text analysis of `video_transcription_text` — this will be central to the classification model
4. **Feature correlation analysis**: Quantify the relationship between `author_ban_status` and `claim_status`
5. **Begin feature engineering**: Prepare variables for hypothesis testing and eventual model training
