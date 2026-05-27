# Course 4: Regression Analysis — Simplify Complex Data Relationships
## TikTok — Logistic Regression Model

**Status:** ✅ Complete  
**Scenario:** TikTok Claims Classification  
**Deliverable type:** Logistic regression notebook + PACE document + Executive Summary

---

## Overview

In Course 4, I built a **logistic regression model** to predict whether a TikTok account is verified or not verified, using video characteristics as features. This model acts as a stepping stone toward the final claims classification model — understanding what predicts verified status helps the team understand how video characteristics relate to the claim/opinion distinction.

---

## Why Logistic Regression?

`verified_status` is a **binary categorical variable** (verified / not verified). Linear regression predicts continuous values, which is inappropriate for binary outcomes. Logistic regression predicts the **probability** of belonging to a category — the right tool for this task.

This was confirmed in Course 3: the hypothesis test showed that verified and not verified accounts have significantly different behavior. Logistic regression is the natural next step to model that relationship.

---

## Key Modeling Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Class imbalance | Upsample minority class | Only 6% verified — would bias model to always predict "not verified" |
| Multicollinearity | Remove `video_like_count` | Correlation ~0.86 with `video_view_count` — violates logistic regression assumption |
| Categorical encoding | OneHotEncoder (drop first) | Converts `claim_status` and `author_ban_status` to numeric, avoids dummy variable trap |
| Train/test split | 75% / 25% | Standard split, `random_state=0` for reproducibility |
| Outliers | IQR capping on like count and comment count | Reduces extreme outlier influence without losing rows |

---

## Features Used

| Feature | Type | Notes |
|---------|------|-------|
| `video_duration_sec` | Numeric | ✅ Included |
| `claim_status` | Categorical → encoded | ✅ Included — strongest predictor |
| `author_ban_status` | Categorical → encoded | ✅ Included |
| `video_view_count` | Numeric | ✅ Included |
| `video_share_count` | Numeric | ✅ Included |
| `video_download_count` | Numeric | ✅ Included |
| `video_comment_count` | Numeric | ✅ Included (outliers capped) |
| `video_like_count` | Numeric | ❌ Excluded — multicollinearity |
| `#`, `video_id` | Identifier | ❌ Excluded — not predictive |

---

## Model Results

### Classification Report

| Metric | Verified | Not Verified | Overall |
|--------|---------|-------------|---------|
| Precision | 74% | 64% | 69% |
| Recall | 52% | 82% | 67% |
| F1-score | 61% | 72% | 67% |
| **Accuracy** | — | — | **67%** |

### Confusion Matrix

```
                 Predicted: Not Verified   Predicted: Verified
Actual: Not Verified       2,350 (TN)          2,132 (FP)
Actual: Verified             816 (FN)          3,716 (TP)
```

### Key Model Coefficients

| Feature | Coefficient | Interpretation |
|---------|------------|----------------|
| `claim_status_opinion` | **+1.85** | Strongest predictor — opinions strongly associated with verified accounts |
| `author_ban_status_under scrutiny` | -0.20 | Under scrutiny → lower odds of being verified |
| `video_duration_sec` | +0.0014 | Slight positive association |
| Other features | ~0.000 | Minimal association with verified status |

---

## Key Insights

1. **`claim_status_opinion` is the dominant predictor** — with a coefficient of 1.85, opinion claim status is by far the strongest predictor of an account being verified. This confirms earlier observations and is a critical signal for the final classification model.

2. **Model is acceptable as a baseline** — 67% accuracy and 82% recall for "not verified" is acceptable but not excellent. Logistic regression may not capture the full complexity.

3. **Next steps:** More powerful models (Random Forest, XGBoost) incorporating NLP features from `video_transcription_text` will likely produce significantly better results.

---

## Deliverables

| File | Description |
|------|-------------|
| [TikTok_Course4_Regression.ipynb](TikTok_Course4_Regression.ipynb) | Full logistic regression notebook |
| [PACE_Strategy_Document_Course4.md](PACE_Strategy_Document_Course4.md) | PACE document |
| [TikTok_Course4_Executive_Summary.pdf](TikTok_Course4_Executive_Summary.pdf) | Executive summary for stakeholders |

---

## Python Skills Demonstrated

```python
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Handle class imbalance
data_minority_upsampled = resample(data_minority, replace=True,
                                   n_samples=len(data_majority), random_state=0)

# One-hot encode categorical features
X_encoder = OneHotEncoder(drop='first', sparse_output=False)
X_train_encoded = X_encoder.fit_transform(X_train[['claim_status', 'author_ban_status']])

# Build and train logistic regression
log_clf = LogisticRegression(random_state=0, max_iter=800).fit(X_train_final, y_train_final)

# Evaluate
print(classification_report(y_test_final, y_pred, target_names=['verified', 'not verified']))
```

---

*Previous: [Course 3 — Hypothesis Testing](../course-3-tiktok/) · Next: Course 5 — Machine Learning*
