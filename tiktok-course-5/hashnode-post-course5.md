# Building a Near-Perfect Claims Classifier: Random Forest vs XGBoost on TikTok Data (Google Advanced DA — Course 5)

*This is the fourth and final post in my Google Advanced Data Analytics Certificate series on the TikTok claims classification project. [Part 1: data inspection](https://danieljasme.hashnode.dev/getting-started-with-tiktok-data-pace-framework-and-initial-inspection-google-advanced-da-certificate-course-1). [Part 2: EDA and visualization](https://danieljasme.hashnode.dev/exploring-tiktok-data-with-python-my-eda-approach-google-advanced-da-certificate-course-2). [Part 3: hypothesis testing and logistic regression](https://danieljasme.hashnode.dev). This post covers the final machine learning model.*

---

## The Full Journey So Far

Over four courses, here's what the TikTok data team accomplished:

- **Course 1:** Loaded and inspected the dataset — 19,383 rows, 12 columns, near-balanced 50/50 claim/opinion split
- **Course 2:** EDA revealed that claim videos receive dramatically higher engagement than opinion videos
- **Course 3:** Hypothesis testing confirmed the engagement difference between verified and unverified accounts is statistically significant
- **Course 4:** Logistic regression on `verified_status` — 67% accuracy, useful baseline but not production-ready

Course 5 is the payoff: **the actual model that TikTok would deploy**.

---

## The Business Problem (And Why It's Harder Than It Looks)

TikTok receives millions of user-reported videos every day. Human moderators cannot review all of them. The data team's task: build a model to automatically classify whether a video contains a **claim** (unverified information presented as fact) or an **opinion** (personal belief).

This sounds straightforward. The tricky part is choosing the right objective.

### Choosing the Right Metric: Recall Over Accuracy

There are two ways this model can be wrong:

- **False positive:** Predicts "claim" when it's actually an opinion → video goes to unnecessary human review
- **False negative:** Predicts "opinion" when it's actually a claim → video containing misinformation never gets reviewed

These errors are not equally bad. A false positive wastes a moderator's time. A false negative lets potentially harmful content go unchecked.

**Decision: optimize for recall.** Minimize false negatives. It's acceptable to over-flag some opinions if it means catching nearly all claims.

This is an ethical judgment, not just a technical one — and it shapes every subsequent decision in the modeling process.

---

## The Ethical Questions First

Before writing a single line of model code, the project required thinking through the ethical implications. This is something data courses often skip, but it matters enormously in production:

**What happens if the model makes errors at scale?**

At TikTok's volume, even a 1% error rate means thousands of incorrect classifications per day. A false negative rate of 1% could mean thousands of violating videos going unreviewed. That's why recall was chosen — not because it makes the technical problem easier, but because the business consequences of that error type are unacceptable.

**Could the model be biased?**

The feature importance analysis (spoiler: engagement metrics dominate) raises a real question: are claim videos amplified by TikTok's algorithm, or do they generate organic engagement? If the platform's own recommendation system pushes claim content to more viewers, then engagement is partly a product of TikTok's choices — not just the content creator's. The model would then be learning a pattern that TikTok itself partially creates.

This doesn't make the model wrong, but it's worth flagging for the team. Regular monitoring for disproportionate flagging of specific creator communities is recommended.

---

## Feature Engineering

Two new things were added at this stage compared to previous courses:

### 1. Text length

```python
data['text_length'] = data['video_transcription_text'].str.len()
data[['claim_status', 'text_length']].groupby('claim_status').mean()

# claim      95.5 characters
# opinion    82.3 characters
```

Claims average about 13 more characters than opinions. Not a massive difference, but a consistent one. Claim videos tend to include more specific (often false) details.

### 2. NLP n-gram features via CountVectorizer

```python
from sklearn.feature_extraction.text import CountVectorizer

count_vec = CountVectorizer(ngram_range=(2, 3),
                            max_features=15,
                            stop_words='english')
```

This converts the transcription text into numerical features by finding the 15 most commonly occurring 2-word and 3-word phrases across the dataset. Phrases like "learned media", "read online", "social media claim" become binary features — does this video's transcript contain this phrase?

**Important rule:** The vectorizer is fitted on training data only, then used to transform validation and test data. Fitting on the full dataset would leak information from the test set into the model.

---

## The 60/20/20 Split

Course 4 used a simple 75/25 train/test split. Course 5 adds a **validation set** — giving us three distinct datasets:

```
19,182 samples total
├── Training (60% = ~11,508) → model fitting + GridSearchCV
├── Validation (20% = ~3,837) → model selection between RF and XGBoost
└── Test (20% = ~3,837) → final evaluation, touched once
```

The test set is held out until the very end. If you look at your test set during model development, you'll unconsciously make decisions that optimize for it — and your reported performance will be optimistic. The validation set is what you use to make decisions.

---

## Building Two Models

### Model 1: Random Forest

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

rf = RandomForestClassifier(random_state=0)

cv_params = {
    'max_depth': [5, 7, None],
    'max_features': [0.3, 0.6],
    'max_samples': [0.7],
    'min_samples_leaf': [1, 2],
    'min_samples_split': [2, 3],
    'n_estimators': [75, 100, 200],
}

rf_cv = GridSearchCV(rf, cv_params,
                     scoring=['accuracy', 'precision', 'recall', 'f1'],
                     cv=5, refit='recall')
rf_cv.fit(X_train_final, y_train)
```

`refit='recall'` tells GridSearchCV to select the best model based on recall score, not accuracy. This is the metric decision made at the start flowing through to the implementation.

**Cross-validation results:**
- Best recall: **99.52%**
- Best precision at that index: **99.98%**

The high precision alongside high recall tells us the model is not just labeling everything as a claim — it's genuinely discriminating between the two classes.

### Model 2: XGBoost

```python
from xgboost import XGBClassifier

xgb = XGBClassifier(objective='binary:logistic', random_state=0)

cv_params = {
    'max_depth': [4, 8, 12],
    'min_child_weight': [3, 5],
    'learning_rate': [0.01, 0.1],
    'n_estimators': [300, 500]
}

xgb_cv = GridSearchCV(xgb, cv_params, scoring=scoring, cv=5, refit='recall')
xgb_cv.fit(X_train_final, y_train)
```

XGBoost cross-validation recall: **98.98%** — strong, but slightly lower than Random Forest.

---

## Model Selection on the Validation Set

| Model | Recall (val) | Precision (val) | Accuracy (val) | FN | FP |
|-------|-------------|----------------|----------------|----|----|
| Random Forest | **99.6%** | 100% | **99.8%** | 7 | 0 |
| XGBoost | 99.1% | 99.9% | 99.5% | 17 | 1 |

Random Forest produced only 7 false negatives on the 3,837-sample validation set. XGBoost had 17. Since recall was the priority metric, **Random Forest is the champion model**.

---

## Final Evaluation on the Test Set

The champion model was evaluated on the held-out test set exactly once:

```
              Predicted: Opinion   Predicted: Claim
Actual: Opinion      1,909 (TN)         1 (FP)
Actual: Claim          9 (FN)       1,918 (TP)
```

- **Test accuracy: 99.7%**
- **Test recall: 99.5%**
- **Test precision: 99.9%**

10 misclassifications out of 3,837 samples.

---

## What's Driving the Model: Feature Importances

```python
importances = rf_cv.best_estimator_.feature_importances_
rf_importances = pd.Series(importances, index=X_test_final.columns)
rf_importances.sort_values(ascending=False).head()

# video_view_count        0.457
# video_share_count       0.162
# video_like_count        0.150
# video_download_count    0.145
# video_comment_count     0.055
```

**Engagement metrics account for ~97% of total feature importance.** Text features, verified status, ban status, and video duration contribute almost nothing individually.

This is both the model's strength and its most interesting finding: the model doesn't actually read the content of the videos. It classifies them based on how many people watched, shared, liked, and downloaded them. Claim videos behave fundamentally differently from opinion videos in terms of engagement — and that signal alone is enough for 99.7% accuracy.

---

## Why Such High Accuracy?

When a model performs this well, it's worth asking: is something wrong? A few sanity checks:

1. **The validation and test sets were held out** — no data leakage from these during training
2. **Precision is also very high** — the model isn't just predicting "claim" for everything
3. **The engagement gap is genuinely enormous** — claim videos get ~47x more views than opinion videos. The signal is real and strong.

The answer is that claim content on TikTok behaves so differently from opinion content — in terms of how users engage with it — that even a simple model should be able to separate them. The Random Forest found this pattern and learned it almost perfectly.

---

## Recommendations for TikTok

1. **Deploy the model** — 99.7% accuracy with 99.5% recall is production-ready
2. **Workflow:** Claims → ranked by report count → top X% reviewed by human moderators daily
3. **Add missing features** in future versions: number of times each video was reported; total reports across all of an author's videos
4. **Monitor quarterly** — content patterns evolve, and the model should be retrained periodically
5. **Maintain human review** — the model triages, it does not make final moderation decisions

---

## What the Full 5-Course Project Taught Me

Looking back across all five courses:

The most important lesson isn't a technical one. It's that **every decision in a data project connects to every other decision**. Choosing recall as the evaluation metric in Course 5 was directly justified by the ethical analysis at the start of the same course. The feature engineering decisions in Course 5 built directly on the EDA patterns found in Course 2. The near-perfect model performance in Course 5 explains why the logistic regression in Course 4 was "good enough as a baseline but not for production."

Data science isn't a sequence of isolated techniques — it's a continuous conversation between the data, the business problem, and the choices you make at each stage.

---

## Portfolio Links

- 📁 **GitHub:** [github.com/danieljasme-analyst/google-advanced-data-analytics-portfolio](https://github.com/danieljasme-analyst/google-advanced-data-analytics-portfolio)
- 📝 **Course 1:** [Initial data inspection](https://danieljasme.hashnode.dev/getting-started-with-tiktok-data-pace-framework-and-initial-inspection-google-advanced-da-certificate-course-1)
- 📝 **Course 2:** [EDA and visualization](https://danieljasme.hashnode.dev/exploring-tiktok-data-with-python-my-eda-approach-google-advanced-da-certificate-course-2)
- 📝 **Courses 3 & 4:** [Hypothesis testing and logistic regression](https://danieljasme.hashnode.dev)

---

*Tags: python, machine-learning, random-forest, xgboost, google-certificate, data-analytics, sklearn, tiktok, portfolio, feature-engineering*
