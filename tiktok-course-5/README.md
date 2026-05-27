# Course 5: The Nuts and Bolts of Machine Learning
## TikTok — Random Forest Claims Classifier

**Status:** ✅ Complete  
**Scenario:** TikTok Claims Classification  
**Deliverable type:** Random Forest + XGBoost notebook + PACE document + Executive Summary

---

## Overview

Course 5 is the culmination of the TikTok claims classification project — building the **final production-ready machine learning model**. Two tree-based classifiers (Random Forest and XGBoost) were developed, tuned with GridSearchCV optimizing for recall, and evaluated on a held-out test set. The Random Forest model was selected as the champion.

---

## Why Recall as the Evaluation Metric?

| Error type | Consequence | Severity |
|-----------|------------|---------|
| False negative (claim predicted as opinion) | Video violates TOS but never reviewed | 🔴 Severe |
| False positive (opinion predicted as claim) | Video goes to unnecessary human review | 🟡 Minor |

Missing actual violations is far worse than over-flagging — **recall was chosen as the primary metric**.

---

## Modeling Approach

### Data Split: 60/20/20
```
All data (19,182)
├── Training (60%) → model fitting + GridSearchCV
├── Validation (20%) → model selection
└── Test (20%) → final evaluation (held out)
```

### Feature Engineering
- `text_length` — character count of `video_transcription_text`
- `CountVectorizer` — 15 most frequent 2-grams and 3-grams from transcription text
- Dummy encoding for `verified_status` and `author_ban_status`
- Target encoding: opinion → 0, claim → 1

**Note:** Unlike Course 4 (logistic regression), `video_like_count` was **retained** — tree-based models handle multicollinearity naturally.

---

## Model Results

### Random Forest (Champion)

| Set | Recall | Precision | Accuracy |
|-----|--------|----------|---------|
| Cross-validation | **99.5%** | **99.98%** | — |
| Validation | **99.6%** | **100%** | **99.8%** |
| **Test** | **99.5%** | **99.9%** | **99.7%** |

**Test confusion matrix:**
```
              Predicted: Opinion   Predicted: Claim
Actual: Opinion      1,909 (TN)         1 (FP)
Actual: Claim          9 (FN)       1,918 (TP)
```
Only 10 misclassifications out of 3,837 test samples.

### XGBoost

| Set | Recall | Precision | Accuracy |
|-----|--------|----------|---------|
| Validation | 99.1% | 99.9% | 99.5% |

**Champion decision:** Random Forest — higher recall (99.6% vs 99.1%) on validation set.

---

## Feature Importances (Champion Model)

| Feature | Importance | Note |
|---------|-----------|------|
| `video_view_count` | **45.7%** | Strongest signal by far |
| `video_share_count` | 16.2% | Engagement |
| `video_like_count` | 15.0% | Engagement |
| `video_download_count` | 14.5% | Engagement |
| `video_comment_count` | 5.5% | Engagement |
| All other features | <3% combined | Minimal contribution |

**Key insight:** The model classifies videos almost entirely based on engagement levels. Claim videos receive dramatically more views, shares, likes, and downloads — the model learned this signal naturally.

---

## Key Python Code

```python
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer

# 60/20/20 split
X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_tr, y_tr, test_size=0.25, random_state=0)

# NLP features
count_vec = CountVectorizer(ngram_range=(2, 3), max_features=15, stop_words='english')
count_data = count_vec.fit_transform(X_train['video_transcription_text']).toarray()

# GridSearchCV tuning for recall
rf_cv = GridSearchCV(RandomForestClassifier(random_state=0), cv_params,
                     scoring=['accuracy','precision','recall','f1'],
                     cv=5, refit='recall')
rf_cv.fit(X_train_final, y_train)

# Best recall: 0.9952
# Best precision: 0.9998
```

---

## Ethical Considerations

1. **Transparency:** Users whose videos are flagged should have access to an appeal process
2. **Bias risk:** Regular monitoring for disproportionate flagging of specific creator communities
3. **Human oversight:** The model triages — it does not make final moderation decisions
4. **Next steps in workflow:** Classified claims → ranked by report count → top X% reviewed daily by human moderators

---

## Deliverables

| File | Description |
|------|-------------|
| [TikTok_Course5_ML.ipynb](TikTok_Course5_ML.ipynb) | Complete ML notebook — RF + XGBoost |
| [PACE_Strategy_Document_Course5.md](PACE_Strategy_Document_Course5.md) | Full PACE document including ethical analysis |
| [TikTok_Course5_Executive_Summary.pdf](TikTok_Course5_Executive_Summary.pdf) | Executive summary for leadership |

---

*Previous: [Course 4 — Logistic Regression](../course-4-tiktok/) · Next: Capstone Project*
