# PACE Strategy Document — TikTok Course 4
## Regression Analysis: Simplify Complex Data Relationships

**Data Professional:** Ahmad Daniel  
**Project:** TikTok Claims Classification  
**Course:** 4 — Regression Analysis  
**Date:** May 2026

---

## PACE: Plan

### Understand the situation

**What is the context of this stage?**  
The TikTok data team observed in Course 3 that verified users are much more likely to post opinions than claims. Leadership now wants to understand what video characteristics predict whether a user is verified. A logistic regression model with `verified_status` as the outcome variable will help the team understand these relationships before building the final claims classification model.

**Why logistic regression and not linear regression?**  
The outcome variable `verified_status` is binary (verified / not verified) — a categorical outcome. Linear regression predicts a continuous numeric value, which is not appropriate here. Logistic regression predicts the probability of belonging to a category, which is exactly what is needed.

**What are the model assumptions for logistic regression?**
1. Binary outcome variable ✅ — verified vs not verified
2. Independence of observations ✅ — each row is a separate video
3. No severe multicollinearity among features — must check with correlation heatmap
4. Large enough sample size ✅ — 19,182 rows after dropping nulls

**Who are the stakeholders?**

| Stakeholder | Role | Communication need |
|-------------|------|--------------------|
| Maika Abadi | Operations Lead | Wanted clarity on regression type — non-technical |
| Rosie Mae Bradshaw | Data Science Manager | Specified deliverables: confusion matrix + accuracy + executive summary |
| Willow Jaffey | Data Science Lead | Reviews before sharing with client |
| Mary Joanna Rodgers | PMO | Final recipient of executive summary |

---

## PACE: Analyze

### EDA findings

**Missing values:** 200 rows dropped (same as previous courses)  
**Duplicates:** 0 duplicate rows found

**Outlier treatment (IQR capping):**
- `video_like_count`: capped at Q3 + 1.5×IQR upper limit
- `video_comment_count`: capped at Q3 + 1.5×IQR upper limit
- Other engagement metrics: kept as-is for this analysis

**Class imbalance issue:**
- Not verified: ~94% of dataset
- Verified: ~6% of dataset
- This severe imbalance would cause a naive model to always predict "not verified" and still achieve 94% accuracy — misleading and useless
- Solution: **Upsample the minority class** (verified) to match the majority class size using `sklearn.utils.resample`
- After upsampling: 18,027 not verified + 18,027 verified = 36,054 total rows

**Multicollinearity check:**
- `video_view_count` and `video_like_count` are strongly correlated (~0.86)
- Decision: **Exclude `video_like_count`** from features to meet the no-multicollinearity assumption
- Kept features: `video_duration_sec`, `claim_status`, `author_ban_status`, `video_view_count`, `video_share_count`, `video_download_count`, `video_comment_count`

**Text length feature:**
- Added `text_length` column: length of `video_transcription_text` in characters
- Verified accounts have slightly longer transcription text on average

---

## PACE: Construct

### Model design

**Outcome variable (Y):** `verified_status` (binary: verified / not verified)

**Features (X):** 7 features selected
- `video_duration_sec` — numeric
- `claim_status` — categorical → one-hot encoded (drop first)
- `author_ban_status` — categorical → one-hot encoded (drop first)
- `video_view_count` — numeric
- `video_share_count` — numeric
- `video_download_count` — numeric
- `video_comment_count` — numeric

**Excluded features:**
- `#` and `video_id` — identifiers, not predictive
- `video_like_count` — strongly correlated with `video_view_count` (multicollinearity)
- `video_transcription_text` — text data, requires NLP pipeline (future work)
- `text_length` — derived from transcription text, excluded for simplicity

**Train/test split:** 75% train / 25% test, `random_state=0`

**Encoding:**
- `claim_status` and `author_ban_status`: `OneHotEncoder(drop='first')` — drops first category to avoid dummy variable trap
- `verified_status` (outcome): same `OneHotEncoder` approach

**Model:** `LogisticRegression(random_state=0, max_iter=800)`

### Feature encoding result

After one-hot encoding, 8 total features in the model:
- `video_duration_sec`, `video_view_count`, `video_share_count`, `video_download_count`, `video_comment_count`
- `claim_status_opinion` (1 = opinion, 0 = claim)
- `author_ban_status_banned` (1 = banned, 0 = otherwise)
- `author_ban_status_under scrutiny` (1 = under scrutiny, 0 = otherwise)

---

## PACE: Execute

### Model results

**Confusion matrix results:**

| | Predicted: Not Verified | Predicted: Verified |
|---|---|---|
| Actual: Not Verified | 2,350 (TN) | 2,132 (FP) |
| Actual: Verified | 816 (FN) | 3,716 (TP) |

**Classification report:**

| Metric | Verified | Not Verified | Weighted Avg |
|--------|---------|-------------|-------------|
| Precision | 74% | 64% | 69% |
| Recall | 52% | 82% | 67% |
| F1-score | 61% | 72% | 67% |
| Accuracy | — | — | **67%** |

**Model coefficients (key findings):**

| Feature | Coefficient | Interpretation |
|---------|------------|----------------|
| `claim_status_opinion` | +1.85 | Strongest predictor — opinion videos strongly associated with verified accounts |
| `video_duration_sec` | +0.00142 | Slightly longer videos → slightly higher odds of verified |
| `author_ban_status_under scrutiny` | -0.20 | Under scrutiny → lower odds of verified |
| Other features | ~0.00 | Very small association with verified status |

### Interpretation for stakeholders

**Key takeaways:**
1. The logistic regression model achieved 67% overall accuracy with 82% recall for the "not verified" class — recall is the more important metric here since it tells us how many "not verified" cases the model correctly identifies
2. The strongest predictor of verified status is `claim_status_opinion` — verified users are significantly more likely to post opinions than claims
3. Other video metrics (view count, share count, download count, comment count) have very small coefficients — their relationship with verified status is minimal
4. The model has acceptable but not excellent predictive power — logistic regression may not fully capture the complexity of this classification problem

**Recommended next steps:**
1. Consider more sophisticated models (Random Forest, XGBoost) for the final claims classification model
2. Incorporate NLP features from `video_transcription_text` — text content is likely a stronger predictor than engagement metrics
3. This logistic regression model serves as a useful baseline for the claims classification project
