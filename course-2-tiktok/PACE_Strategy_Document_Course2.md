# PACE Strategy Document — TikTok Course 2
## Get Started with Python: Exploratory Data Analysis

**Data Professional:** Ahmad Daniel  
**Project:** TikTok Claims Classification  
**Course:** 2 — Get Started with Python  
**Date:** May 2026

---

## What is PACE?

| Stage | Description | Course 2 Application |
|-------|-------------|----------------------|
| **P**lan | Define scope, understand the situation | Review stakeholder requests, plan EDA approach |
| **A**nalyze | Explore and clean data | Load data, check nulls, examine distributions |
| **C**onstruct | Build visualizations | Create matplotlib/seaborn charts + Tableau dashboard |
| **E**xecute | Share results | Executive summary for data team and stakeholders |

---

## PACE: Plan

### Understand the situation

**What is the context of this project?**  
The TikTok data team has completed initial data inspection (Course 1) and is now ready to conduct formal Exploratory Data Analysis (EDA). The goal is to understand the data's structure, identify patterns, detect outliers, and create visualizations that communicate findings to both technical and non-technical stakeholders.

**What are the stakeholder requests for this stage?**

From **Orion Rainier (Data Scientist):**
- A Python notebook showing data structuring, cleaning, and matplotlib/seaborn visualizations
- At minimum: claim vs opinion count comparison, boxplots of key variables (video duration, video like count, video comment count, video view count), and author ban status breakdown
- A Tableau dashboard with scatter plot (views vs likes), claim/opinion count, and author status — for non-technical stakeholder communication
- Dashboard must be accessible to someone with visual impairments

From **Willow Jaffey (Data Science Lead):**
- Executive summary of EDA findings attached via email

**Who are the stakeholders and how do I communicate with them?**

| Stakeholder | Role | Communication style |
|-------------|------|---------------------|
| Orion Rainier | Data Scientist — technical | Concise, specific, technical detail welcome |
| Rosie Mae Bradshaw | Data Science Manager | High-level progress updates |
| Willow Jaffey | Data Science Lead | Executive summary format |
| Mary Joanna Rodgers | Project Management Officer | Non-technical; milestone updates |

**What questions need to be answered during EDA?**
- How are claims and opinions distributed? Is the dataset balanced for modeling?
- Are there null values? How should they be handled?
- What do the distributions of engagement variables look like — normal or skewed?
- Are there significant outliers in view counts, like counts, or comment counts?
- How do engagement metrics differ between claims and opinions?
- Is there a relationship between author ban status and claim status?

**What are the ethical considerations?**
- The dataset is synthetic and created for educational purposes — no real user privacy concerns
- When building the future model, care must be taken not to create bias against certain author groups based on ban status alone
- Accessibility: Tableau dashboard must use color combinations that work for users with visual impairments (color + shape encoding, not color alone)

---

## PACE: Analyze

### Data exploration notes

**Dataset:** `tiktok_dataset.csv` — 19,382 rows × 12 columns

**Null value findings:**
- 200 rows have null values across 7 columns simultaneously: `claim_status`, `video_transcription_text`, and all 5 engagement columns
- These appear to be systematic nulls (all missing together) rather than random — likely a data collection failure for these specific videos
- Decision: **Drop null rows for EDA** (200/19,382 = 1.03% — negligible loss)
- For modeling: investigate further before deciding on permanent treatment

**Duplicate check:**
- 0 duplicate rows found
- 0 duplicate `video_id` values found

**Claim status distribution (after dropping nulls):**
- Claims: 9,670 (50.4%)
- Opinions: 9,512 (49.6%)
- Assessment: Near-balanced ✅ — favorable for unbiased ML classification

**Author ban status distribution:**
- Active: 15,604 (81.3%)
- Under scrutiny: 2,149 (11.2%)
- Banned: 1,629 (8.5%)
- Key observation: Banned authors are disproportionately associated with claim videos — this relationship should be explored further

**Engagement variable distributions:**

| Variable | Min | Max | Mean | Median | Distribution |
|----------|-----|-----|------|--------|-------------|
| `video_duration_sec` | 5 | 60 | 32.5 | 32 | Roughly uniform |
| `video_view_count` | 21 | 999,976 | 253,763 | ~75,000 | Heavy right skew |
| `video_like_count` | 0 | 656,081 | 83,235 | ~25,000 | Heavy right skew |
| `video_comment_count` | 0 | 7,887 | 342 | ~90 | Heavy right skew |

**Outlier assessment:**
- Engagement metrics all show extreme right-skewed distributions with significant outliers
- The majority of videos have fewer than 100,000 views, fewer than 100,000 likes, and fewer than 100 comments
- A small number of viral videos drive the means far above the medians
- Video duration is the only variable without significant outlier concerns

**Key pattern — claim vs opinion engagement:**
- Claim videos have dramatically higher view counts, like counts, and comment counts than opinion videos
- This pattern holds across all author ban status categories
- Engagement metrics will likely be strong predictive features for the classification model

---

## PACE: Construct

### Visualization plan

**Python visualizations (matplotlib/seaborn):**

| Chart | Type | Purpose |
|-------|------|---------|
| Claim vs Opinion Count | Bar chart | Show near-balanced target distribution |
| Author Ban Status | Bar chart | Show distribution of ban categories |
| Key Variables by Claim Status | 4 boxplots | Check for outliers and claim/opinion differences |
| View Count Histogram | Histogram | Show right-skewed distribution |
| Like Count Histogram | Histogram | Show right-skewed distribution |
| Comment Count Histogram | Histogram | Show right-skewed distribution |
| Claim by Ban Status | Grouped bar | Show relationship between ban status and claim status |

**Tableau dashboard plan:**

*Dashboard 1 — Main stakeholder view:*
- Scatter plot: `video_view_count` vs `video_like_count` colored by `claim_status` (required deliverable)
- Pie/bar chart: claim vs opinion count
- Bar chart: author ban status distribution

*Dashboard 2 — Boxplot view:*
- Boxplots for video duration, like count, comment count, view count

*Accessibility considerations:*
- Use both color AND shape encoding on scatter plot (circle = claim, square = opinion)
- Include descriptive chart titles and axis labels
- Avoid red/green only combinations — use blue/orange palette

### Outlier treatment strategy

| Approach | When to use | Pros | Cons |
|----------|------------|------|------|
| Keep as-is | EDA phase | See full distribution | Skews statistics |
| Log transform | Modeling phase | Normalizes skewed distributions | Loses interpretability |
| IQR capping | Modeling phase | Reduces outlier influence, keeps rows | May lose important signals |
| Row removal | Last resort only | Clean distribution | Loses data, potential bias |

**Recommended:** Log-transform engagement variables for modeling. Do NOT drop outlier rows — high-engagement videos may be disproportionately claims and removing them would introduce bias.

---

## PACE: Execute

### Summary findings for stakeholders

**What percentage of data is claims vs opinions?**  
50.4% claims (9,670) and 49.6% opinions (9,512) — near-balanced, no resampling needed for initial modeling.

**What factors correlate with a video's claim status?**  
- All engagement metrics (views, likes, shares, downloads, comments) are higher for claim videos
- Banned and under-scrutiny authors post disproportionately more claims
- Video duration does NOT correlate with claim status

**How should outliers be handled?**  
Log-transform skewed engagement variables for modeling. Keep all rows including high-engagement outliers as they may carry important classification signal.

**What are the recommended next steps?**
1. Create Tableau dashboard for non-technical stakeholder communication
2. Conduct formal statistical hypothesis testing to confirm observed differences are significant
3. Begin NLP feature extraction from `video_transcription_text`
4. Investigate null row pattern before modeling phase
