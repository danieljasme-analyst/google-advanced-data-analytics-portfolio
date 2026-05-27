# From Hypothesis Testing to Logistic Regression: TikTok Claims Project (Google Advanced DA — Courses 3 & 4)

*This is the third post in my Google Advanced Data Analytics Certificate series, documenting the TikTok claims classification project. [Part 1 covers initial data inspection (Course 1)](https://danieljasme.hashnode.dev/getting-started-with-tiktok-data-pace-framework-and-initial-inspection-google-advanced-da-certificate-course-1). [Part 2 covers EDA and visualization (Course 2)](https://danieljasme.hashnode.dev/exploring-tiktok-data-with-python-my-eda-approach-google-advanced-da-certificate-course-2). This post covers statistical hypothesis testing (Course 3) and logistic regression modeling (Course 4).*

---

## Where We Left Off

After two courses of exploring and visualizing the TikTok dataset, the patterns were clear:

- Claim videos get dramatically more views, likes, and shares than opinion videos
- Not verified accounts post significantly more claims than verified accounts
- Engagement metrics are heavily right-skewed — a small number of viral videos distort the averages

The questions in Courses 3 and 4 are: **how do we turn these observations into statistically provable statements, and then into an actual predictive model?**

---

## Course 3 — Hypothesis Testing: Proving the Difference is Real

### The Business Request

TikTok's Project Management Officer, Mary Joanna Rodgers, raised a specific question to the data team: *is there a statistically significant difference in video view counts between verified and unverified accounts?*

This is an important distinction. In EDA, we can see that the means look different. But "looks different" isn't the same as "is actually different." Sampling variation means that even two identical populations would produce different sample means. Hypothesis testing gives us a formal framework for deciding whether the observed difference is real or just noise.

### Setting Up the Test

**Research question:** Do verified and not verified TikTok accounts have different mean video view counts?

**Hypotheses:**
- **H₀ (null):** There is no difference in mean video view count between verified and not verified accounts
- **Hₐ (alternative):** There IS a statistically significant difference

**Why Welch's two-sample t-test?**
- Two independent groups: verified vs not verified
- One continuous dependent variable: `video_view_count`
- Welch's variant handles unequal variances between groups — important here since the standard deviations differ enormously (221K for verified vs 323K for not verified)
- Significance level: α = 0.05

```python
from scipy import stats

not_verified = data[data["verified_status"] == "not verified"]["video_view_count"]
verified = data[data["verified_status"] == "verified"]["video_view_count"]

stats.ttest_ind(a=not_verified, b=verified, equal_var=False)
```

### The Result — and Why It's Counterintuitive

| Group | Sample size | Mean views | Median views |
|-------|------------|-----------|-------------|
| Verified | 1,155 | 88,809 | 6,134 |
| Not verified | 18,027 | 264,332 | 46,415 |

**t-statistic: 25.29 · p-value: ~0.000 → Reject H₀**

The difference is statistically significant — not verified accounts get roughly 3x more views on average than verified accounts. At first glance, this seems backwards. Shouldn't more credible, verified accounts attract more viewers?

The explanation connects back to what we found in Course 2: not verified accounts post more claims, and claim videos generate dramatically higher engagement. So it's not that being unverified causes more views — it's that unverified accounts are more likely to post the kind of content (claims, sensational information) that drives viral engagement.

**This distinction matters enormously for modeling.** Statistical significance tells us the difference is real, not why it exists. Building a model on this relationship without understanding the mechanism could lead to biased predictions.

---

## Course 4 — Logistic Regression: Building the First Model

### Why Logistic Regression?

The hypothesis test in Course 3 confirmed that verified status relates to video behavior. The natural next step is to model that relationship — and specifically, to predict whether an account is verified based on video characteristics.

Why logistic regression and not linear regression? Because `verified_status` is **binary** — verified or not verified. Linear regression predicts a continuous number (like revenue or temperature), which makes no sense for a yes/no outcome. Logistic regression predicts the **probability** of belonging to a category, which is exactly what we need.

### The Critical Preprocessing Step: Handling Class Imbalance

Before touching the model, there's a fundamental data problem to solve.

```python
data["verified_status"].value_counts(normalize=True)
# not verified    0.94
# verified        0.06
```

Only 6% of accounts are verified. If we trained a model on this raw distribution, a naive classifier could achieve 94% accuracy by simply always predicting "not verified" — without learning anything meaningful. This is one of the most common traps in classification modeling.

The fix: **upsample the minority class** (verified) to match the majority class size.

```python
from sklearn.utils import resample

data_majority = data[data["verified_status"] == "not verified"]
data_minority = data[data["verified_status"] == "verified"]

data_minority_upsampled = resample(data_minority,
                                   replace=True,
                                   n_samples=len(data_majority),
                                   random_state=0)

data_upsampled = pd.concat([data_majority, data_minority_upsampled]).reset_index(drop=True)
# Result: 18,027 not verified + 18,027 verified = 36,054 rows
```

### Removing Multicollinear Features

One of logistic regression's core assumptions is **no severe multicollinearity** — features shouldn't be highly correlated with each other. The correlation heatmap revealed that `video_view_count` and `video_like_count` have a correlation coefficient of ~0.86.

Decision: **drop `video_like_count`** from the feature set.

The final 7 features selected:
- `video_duration_sec`
- `claim_status` (categorical → one-hot encoded)
- `author_ban_status` (categorical → one-hot encoded)
- `video_view_count`
- `video_share_count`
- `video_download_count`
- `video_comment_count` (outliers capped at IQR upper limit)

### Building the Model

```python
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Encode categorical features
X_encoder = OneHotEncoder(drop='first', sparse_output=False)
X_train_encoded = X_encoder.fit_transform(X_train[['claim_status', 'author_ban_status']])

# Train/test split: 75/25
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Build model
log_clf = LogisticRegression(random_state=0, max_iter=800).fit(X_train_final, y_train_final)
```

### Results

**Classification report:**

| Metric | Verified | Not Verified | Overall |
|--------|---------|-------------|---------|
| Precision | 74% | 64% | 69% |
| Recall | 52% | 82% | 67% |
| F1-score | 61% | 72% | 67% |
| Accuracy | — | — | **67%** |

**Confusion matrix:**
```
                   Predicted: Not Verified   Predicted: Verified
Actual: Not Verified      2,350 (TN)             2,132 (FP)
Actual: Verified            816 (FN)             3,716 (TP)
```

67% accuracy. Is that good? It depends on context. For this baseline model, it's acceptable but not excellent. The 82% recall for "not verified" is the more encouraging number — the model correctly identifies 82% of unverified accounts.

### The Most Important Finding: Model Coefficients

```python
pd.DataFrame({
    "Feature Name": log_clf.feature_names_in_,
    "Model Coefficient": log_clf.coef_[0]
})
```

| Feature | Coefficient |
|---------|------------|
| `claim_status_opinion` | **+1.85** |
| `author_ban_status_under scrutiny` | -0.20 |
| `video_duration_sec` | +0.0014 |
| All other features | ~0.000 |

`claim_status_opinion` has a coefficient of **+1.85** — by far the largest in the model. Opinion videos are strongly associated with verified accounts. Every other feature has a coefficient close to zero, meaning their individual relationship with verified status is minimal after controlling for the others.

This confirms what we've seen across three courses: **claim status is the central variable in this dataset**. Everything else — engagement metrics, ban status, video duration — relates to claim status, which is ultimately what predicts user behavior.

---

## Connecting the Two Courses

Courses 3 and 4 tell a connected story:

**Course 3:** We proved statistically that verified and not verified accounts behave differently (in terms of view counts). The difference is real, not random.

**Course 4:** We quantified *how* video characteristics relate to verified status. The answer: almost entirely through claim status. Opinion videos are what makes an account "verified-like."

This matters for the final model (Course 5): if we want to classify videos as claims or opinions, verified status and engagement metrics are useful signals — but they're largely proxies for the underlying content type. The transcription text (NLP features) will likely be the most direct signal.

---

## What I'd Do Differently

**In Course 3:** I would also run a chi-square test on the relationship between `verified_status` and `claim_status` directly. The t-test on view counts shows behavioral differences, but testing the categorical relationship between these two variables would have been a cleaner way to prove what we already suspected.

**In Course 4:** The logistic regression model is a useful baseline, but I'd want to test tree-based models (Random Forest, XGBoost) earlier. They handle class imbalance and non-linear relationships more naturally and would likely outperform logistic regression on this dataset.

---

## Portfolio Links

- 📁 **GitHub:** [github.com/danieljasme-analyst/google-advanced-data-analytics-portfolio](https://github.com/danieljasme-analyst/google-advanced-data-analytics-portfolio)
- 📝 **Course 1:** [Initial data inspection](https://danieljasme.hashnode.dev/getting-started-with-tiktok-data-pace-framework-and-initial-inspection-google-advanced-da-certificate-course-1)
- 📝 **Course 2:** [EDA and visualization](https://danieljasme.hashnode.dev/exploring-tiktok-data-with-python-my-eda-approach-google-advanced-da-certificate-course-2)

---

*Tags: python, statistics, machine-learning, logistic-regression, hypothesis-testing, google-certificate, tiktok, sklearn, pandas, portfolio*
