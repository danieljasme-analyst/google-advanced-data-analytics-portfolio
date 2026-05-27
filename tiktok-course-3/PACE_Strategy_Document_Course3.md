# PACE Strategy Document — TikTok Course 3
## The Power of Statistics: Hypothesis Testing

**Data Professional:** Ahmad Daniel  
**Project:** TikTok Claims Classification  
**Course:** 3 — The Power of Statistics  
**Date:** 28 May 2026

---

## PACE: Plan

### Understand the situation

**What is the context of this stage?**  
The TikTok data team has completed the project proposal (Course 1) and EDA with visualizations (Course 2). Leadership has requested a statistical analysis to determine whether there is a meaningful difference in video view counts between verified and unverified accounts.

**What is the specific research question?**  
Is there a statistically significant difference in the mean number of video views for TikTok videos posted by verified accounts versus not verified accounts?

**Who made this request and why?**

| Stakeholder | Request |
|-------------|---------|
| Mary Joanna Rodgers (PMO) | Leadership wants to know if verified status impacts video performance |
| Rosie Mae Bradshaw (Data Science Manager) | Wants two-sample hypothesis test on video view counts by verified status |
| Orion Rainier (Data Scientist) | Technical execution and executive summary before presenting to Willow Jaffey |

**What are the ethical considerations?**
- Results should not be used to discriminate against unverified users
- Verified status should be treated as a correlational signal, not a causal explanation
- The relationship between verified status, claim status, and engagement is complex — oversimplification could lead to biased model decisions

**What type of hypothesis test is appropriate?**  
A **two-sample t-test (Welch's variant)** because:
- Two independent groups: verified vs not verified
- One continuous dependent variable: `video_view_count`
- Welch's variant does not assume equal variances (appropriate here given large variance difference)
- Two-tailed test: testing for difference in either direction

---

## PACE: Analyze

### Data exploration notes

**Dataset after dropping nulls:** 19,182 rows × 12 columns

**Verified status distribution:**

| Status | Count | Percentage |
|--------|-------|-----------|
| not verified | 18,027 | 94.0% |
| verified | 1,155 | 6.0% |

The dataset is heavily imbalanced by verified status — only 6% of accounts are verified. This is typical for social platforms.

**Descriptive statistics for `video_view_count` by verified status:**

| Metric | Verified | Not Verified |
|--------|---------|-------------|
| Count | 1,155 | 18,027 |
| Mean | 88,809 | 264,332 |
| Median | 6,134 | 46,415 |
| Std Dev | 221,179 | 323,334 |
| Min | 21 | 21 |
| Max | 999,976 | 999,976 |

**Key observation:** Not verified accounts have a mean view count approximately **3x higher** than verified accounts. However, both groups show extreme right skew (mean >> median) with very high standard deviations. The Central Limit Theorem allows us to proceed with a t-test given both sample sizes are large (n > 30).

---

## PACE: Construct

### Hypothesis test setup

**Null hypothesis (H₀):**  
There is no difference in the mean video view count between videos posted by verified accounts and those posted by not verified accounts. Any observed difference is due to random sampling variation.

**Alternative hypothesis (Hₐ):**  
There is a statistically significant difference in the mean video view count between videos posted by verified accounts and those posted by not verified accounts.

**Significance level:** α = 0.05

**Test:** Welch's two-sample t-test (two-tailed)

### Assumptions check

| Assumption | Status |
|-----------|--------|
| Independent groups | ✅ Met — verified and not verified are mutually exclusive |
| Continuous dependent variable | ✅ Met — video_view_count is continuous |
| Large enough samples (CLT) | ✅ Met — verified n=1,155, not verified n=18,027 |
| Equal variances not required | ✅ Welch's t-test handles unequal variances |

### Test results

| Result | Value |
|--------|-------|
| t-statistic | -25.2939 |
| p-value | ~0.000000 |
| α | 0.05 |
| Decision | **Reject H₀** |

The p-value is effectively zero — far below the significance level of 0.05. There is overwhelming statistical evidence that the difference in mean video view counts between verified and not verified accounts is not due to random chance.

---

## PACE: Execute

### Interpretation for stakeholders

**Statistical conclusion:**  
We reject the null hypothesis. There is a statistically significant difference in the mean video view count between verified and not verified TikTok accounts (t = -25.29, p < 0.05).

**Business interpretation:**  
Not verified accounts have a substantially higher mean view count (~264,332) compared to verified accounts (~88,809). This counterintuitive finding is likely explained by the relationship between verified status and claim status discovered in Course 2:
- Not verified accounts post more claims
- Claim videos generate dramatically higher engagement
- Higher engagement → higher view counts

**Important limitation:**  
Statistical significance ≠ causation. The difference in views is real but the mechanism behind it involves multiple interacting variables. Verified status alone should not be interpreted as causing lower viewership.

### Recommended next steps

1. Test the relationship between `verified_status` and `claim_status` with a chi-square test
2. Conduct additional t-tests on other engagement metrics (likes, shares, comments)
3. Use regression analysis (Course 4) to model multiple variables simultaneously
4. Proceed with caution when including `verified_status` as a model feature to avoid proxy discrimination
