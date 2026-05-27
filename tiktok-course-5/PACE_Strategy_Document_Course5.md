# PACE Strategy Document — TikTok Course 5
## The Nuts and Bolts of Machine Learning

**Data Professional:** Ahmad Daniel  
**Project:** TikTok Claims Classification  
**Course:** 5 — Machine Learning  
**Date:** 28 May 2026

---

## PACE: Plan

### Ethical considerations

**What are you being asked to do?**  
Build a machine learning model that predicts whether a TikTok video contains a claim or an opinion. This is the final deliverable of the claims classification project — the model that TikTok will use to triage user-reported videos for human review.

**What metric should be used to evaluate success?**  
**Recall** — because it is more important to minimize false negatives (claiming a video is an opinion when it is actually a claim) than false positives.

**Why recall over precision or accuracy?**

| Error type | What happens | Severity |
|-----------|-------------|---------|
| False negative (claim → opinion) | Video violates terms of service but never gets reviewed | 🔴 Severe — harm goes unchecked |
| False positive (opinion → claim) | Video goes to human review unnecessarily | 🟡 Minor — human effort wasted |

Missing actual violations (false negatives) is far worse than over-flagging opinions.

**Ethical implications:**

1. **Transparency:** Users whose videos are flagged should understand why and have an appeal process
2. **Bias risk:** The model may unfairly target certain types of content creators if training data is not representative
3. **Banned authors:** The model should not use author ban status as a direct feature — this could create a self-fulfilling cycle where banned authors' content is always flagged regardless of content
4. **Accountability:** A human review step remains critical — the model should triage, not make final decisions

**Should TikTok proceed with this model?**  
Yes — the model achieves ~99.5% recall on cross-validation with near-perfect precision. It correctly identifies nearly all claim videos while rarely misclassifying opinion videos. The benefits of reducing the human moderation backlog substantially outweigh the small risk of false positives going to unnecessary review.

**Modeling workflow:** 60/20/20 train/validate/test split
1. Fit and tune models on training set (GridSearchCV with 5-fold cross-validation)
2. Select champion model on validation set
3. Final evaluation on test set (held out throughout)

---

## PACE: Analyze

### Data exploration notes

**Dataset:** 19,182 rows × 12 columns (after dropping 200 null rows)

**Class balance:** 50.4% claims / 49.6% opinions — balanced ✅ No resampling needed

**Outliers:** Tree-based models (Random Forest, XGBoost) are robust to outliers — no capping or removal needed unlike logistic regression

**Key new feature engineered:**
- `text_length`: character count of `video_transcription_text`
  - Claim videos average: ~95.5 characters
  - Opinion videos average: ~82.3 characters
  - Claims tend to be slightly longer, consistent with more detailed unverified assertions

**CountVectorizer NLP features:**
- n-grams: 2-grams and 3-grams
- `max_features`: 15 most frequent tokens kept
- `stop_words='english'`: common English words filtered out
- Fit on training data only, transform applied to validation and test

---

## PACE: Construct

### Feature engineering decisions

**Encoding:**
- `claim_status`: replaced with {opinion: 0, claim: 1} → target variable
- `verified_status`: dummy encoded (drop first)
- `author_ban_status`: dummy encoded (drop first)
- `#` and `video_id`: dropped — identifiers, not predictive

**Final feature set (25 total after CountVectorizer):**
- 10 numeric/encoded features: `video_duration_sec`, `video_view_count`, `video_like_count`, `video_share_count`, `video_download_count`, `video_comment_count`, `text_length`, `verified_status_verified`, `author_ban_status_banned`, `author_ban_status_under scrutiny`
- 15 n-gram text features from CountVectorizer

**Why NOT drop `video_like_count` here?**  
Unlike logistic regression (Course 4), tree-based models handle multicollinearity naturally through feature importance — correlated features don't violate model assumptions, so `video_like_count` is retained.

### Model selection rationale

| Model | Why considered | Why/Why not chosen |
|-------|---------------|-------------------|
| Random Forest | Robust to outliers, handles multicollinearity, interpretable feature importances | ✅ Champion — best recall and precision |
| XGBoost | Often outperforms RF on structured data | ❌ Slightly lower recall on validation |
| Logistic Regression | Already built in Course 4 | ❌ 67% accuracy — insufficient for production |
| Naive Bayes | Fast, good for text | ❌ Not tested — team chose tree-based approach |

### Hyperparameter tuning (Random Forest)

| Parameter | Values tested | Best value |
|-----------|--------------|-----------|
| `max_depth` | [5, 7, None] | None (no limit) |
| `max_features` | [0.3, 0.6] | 0.3 |
| `max_samples` | [0.7] | 0.7 |
| `min_samples_leaf` | [1, 2] | 1 |
| `min_samples_split` | [2, 3] | 2 |
| `n_estimators` | [75, 100, 200] | 75 |

---

## PACE: Execute

### Model results

**Random Forest (champion model):**

| Set | Recall | Precision | F1 | Accuracy |
|-----|--------|----------|----|---------|
| Cross-validation (train) | **99.5%** | **99.98%** | — | — |
| Validation | **99.6%** | **100%** | 99.8% | **99.8%** |
| Test | **99.5%** | **99.9%** | 99.7% | **99.7%** |

**XGBoost:**

| Set | Recall | Precision | F1 | Accuracy |
|-----|--------|----------|----|---------|
| Validation | 99.1% | 99.9% | 99.5% | 99.5% |

**Champion model decision:** Random Forest wins — higher recall (99.6% vs 99.1%) on the validation set. Since recall is the primary metric, Random Forest is selected.

### Feature importances (Random Forest test set)

| Feature | Importance | Category |
|---------|-----------|---------|
| `video_view_count` | 45.7% | Engagement |
| `video_share_count` | 16.2% | Engagement |
| `video_like_count` | 15.0% | Engagement |
| `video_download_count` | 14.5% | Engagement |
| `video_comment_count` | 5.5% | Engagement |
| All other features | <3% combined | — |

**Key insight:** The model is almost entirely driven by engagement metrics. Text features, verified status, and ban status contribute very little. Claim videos are identified primarily because they generate dramatically higher engagement — not because of their content.

### Interpretation for stakeholders

1. **Should TikTok use this model?** Yes — 99.7% test accuracy with 99.5% recall. The model is production-ready.

2. **What is the model doing?** Classifying videos primarily based on engagement levels. Claim videos receive dramatically more views, shares, likes, and downloads — the model learned this pattern.

3. **What features would further improve it?** Number of times each video was reported, and total reports for each author's entire video catalog.

4. **Next steps for video review workflow:** Videos classified as claims → ranked by report count → top X% reviewed by human moderators daily. Videos classified as opinions → lower priority queue, sampled for quality control.

5. **Impact of banned authors:** Banned authors disproportionately post claims. However, author ban status contributes minimal importance to the model (~0.5%) — the engagement signal is far stronger.
