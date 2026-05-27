# Course 3: The Power of Statistics
## TikTok — Hypothesis Testing

**Status:** ✅ Complete  
**Scenario:** TikTok Claims Classification  
**Deliverable type:** Hypothesis test notebook + PACE document + Executive Summary

---

## Overview

In Course 3, I conducted formal statistical hypothesis testing on the TikTok dataset. The goal was to determine whether there is a statistically significant difference in video view counts between verified and not verified TikTok accounts — providing leadership with a data-driven answer to a business question.

This stage sits in the **Construct** phase of the PACE framework — using statistical methods to build structured evidence for decision-making.

---

## Business Request

Leadership asked: *"Is there a statistical difference in the data between verified and unverified accounts?"*

The data team focused on `video_view_count` as the key metric, conducting a **two-sample Welch's t-test** to compare verified vs not verified accounts.

---

## Hypothesis Test

| Element | Detail |
|---------|--------|
| **H₀ (Null)** | No difference in mean video view count between verified and not verified accounts |
| **Hₐ (Alternative)** | There IS a statistically significant difference |
| **Test** | Two-sample Welch's t-test (two-tailed) |
| **Significance level** | α = 0.05 |

**Why Welch's t-test?**
- Two independent groups (verified vs not verified)
- Continuous dependent variable (`video_view_count`)
- Welch's variant handles unequal variances (standard deviations differ greatly between groups)
- Large sample sizes → Central Limit Theorem applies

---

## Results

| Metric | Verified | Not Verified |
|--------|---------|-------------|
| Sample size | 1,155 | 18,027 |
| Mean views | 88,809 | 264,332 |
| Median views | 6,134 | 46,415 |
| Std deviation | 221,179 | 323,334 |

| Test Result | Value |
|------------|-------|
| t-statistic | -25.2939 |
| p-value | ~0.000000 |
| Decision | **Reject H₀** ✅ |

---

## Key Findings

**1. The difference is statistically significant**  
The p-value (~0.000) is far below α = 0.05. We reject the null hypothesis — the difference in mean video view counts between verified and not verified accounts is real, not due to random chance.

**2. Not verified accounts get ~3x more views on average**  
Mean view count: 264,332 (not verified) vs 88,809 (verified). Counterintuitive, but likely explained by the claim/opinion dynamic — not verified accounts post more claims, and claim videos drive dramatically higher engagement.

**3. Statistical significance ≠ causation**  
The t-test tells us the difference exists, not why. Verified status, claim status, and engagement are all interrelated. Regression analysis (Course 4) will model these relationships simultaneously.

---

## Deliverables

| File | Description |
|------|-------------|
| [TikTok_Course3_Hypothesis_Testing.ipynb](TikTok_Course3_Hypothesis_Testing.ipynb) | Complete hypothesis testing notebook |
| [PACE_Strategy_Document_Course3.md](PACE_Strategy_Document_Course3.md) | PACE document covering all four stages |
| [TikTok_Course3_Executive_Summary.pdf](TikTok_Course3_Executive_Summary.pdf) | One-page summary for stakeholders |

---

## Python Skills Demonstrated

```python
from scipy import stats

# Separate groups
verified = data[data['verified_status'] == 'verified']['video_view_count']
not_verified = data[data['verified_status'] == 'not verified']['video_view_count']

# Descriptive stats by group
data.groupby('verified_status')['video_view_count'].agg(
    count='count', mean='mean', median='median', std='std'
)

# Welch's two-sample t-test
t_stat, p_value = stats.ttest_ind(verified, not_verified, equal_var=False)
# t-statistic: -25.2939
# p-value: ~0.000000 → Reject H₀
```

---

*Previous: [Course 2 — EDA and Visualization](../course-2-tiktok/) · Next: Course 4 — Regression Analysis*
