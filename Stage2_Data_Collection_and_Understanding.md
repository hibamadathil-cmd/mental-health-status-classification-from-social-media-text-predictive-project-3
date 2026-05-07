# Stage 2: Data Collection & Understanding
## Mental Health Status Classification — Project Pipeline

---

## 1. Data Source

The dataset originates from **Reddit**, one of the largest online community platforms, specifically from mental health–focused subreddits. Reddit provides rich, voluntary, and largely unfiltered self-expression about mental health, making it a widely used source in mental health NLP research.

**Presumed Source Subreddits (by class):**

| Target Label | Subreddit(s) | Condition |
|---|---|---|
| 0 | r/mentalhealth | General / Control |
| 1 | r/depression | Depression |
| 2 | r/Anxiety | Anxiety |
| 3 | r/AvPD, r/bipolar | Avoidant PD / Bipolar |
| 4 | r/SuicideWatch, r/PTSD | Suicidal Ideation / PTSD |

> Exact subreddit-to-label mapping is inferred from content review during Stage 4 (EDA).

---

## 2. Dataset Summary

| Property | Value |
|---|---|
| File | `data_to_be_cleansed.csv` |
| Total Records | **5,957** |
| Features | 4 columns (index, text, title, target) |
| Target Classes | **5** (labels: 0, 1, 2, 3, 4) |
| Language | English |
| Source | Reddit posts (title + body text) |

---

## 3. Column Descriptions

| Column | Type | Description |
|---|---|---|
| `Unnamed: 0` | int | Original row index from data source |
| `title` | str | Title of the Reddit post (always present) |
| `text` | str | Body/content of the Reddit post (can be null) |
| `target` | int | Class label (0–4) representing mental health status |

---

## 4. Class Distribution

The dataset is **well-balanced** across all 5 classes:

| Label | Count | Percentage |
|---|---|---|
| 0 | 1,181 | 19.8% |
| 1 | 1,202 | 20.2% |
| 2 | 1,185 | 19.9% |
| 3 | 1,201 | 20.2% |
| 4 | 1,188 | 19.9% |
| **Total** | **5,957** | **100%** |

**Observation:** Near-perfect class balance (within ±0.4%) — this eliminates the need for oversampling/undersampling techniques (e.g., SMOTE) and allows use of standard accuracy alongside macro-F1 during evaluation.

---

## 5. Missing Values Analysis

| Column | Missing Count | Missing % |
|---|---|---|
| `Unnamed: 0` | 0 | 0.0% |
| `title` | 0 | 0.0% |
| `text` | **350** | **5.9%** |
| `target` | 0 | 0.0% |

### Missing Text by Class

| Class | Missing Text | Total | Missing % |
|---|---|---|---|
| 0 | 82 | 1,181 | 6.9% |
| 1 | 0 | 1,202 | **0.0%** |
| 2 | 100 | 1,185 | 8.4% |
| 3 | 124 | 1,201 | **10.3%** |
| 4 | 44 | 1,188 | 3.7% |

**Key Observations:**
- Class 1 (Depression) has **zero** missing text records
- Class 3 has the highest missing rate (10.3%) — posts where only the title was posted
- These are **title-only posts**, not corrupted entries — the title field is always available as a fallback feature

---

## 6. Text Length Analysis

Analysis performed on the `text` field (NaN treated as empty string, word count = 0):

### Character Length (text field)

| Statistic | Value |
|---|---|
| Mean | 787 characters |
| Median | 553 characters |
| Std Dev | 879 characters |
| Min | 0 (null/empty) |
| Max | 27,542 characters |
| 25th Percentile | 256 characters |
| 75th Percentile | 1,035 characters |

### Word Count by Class

| Class | Mean Words | Median Words | Std Dev | Max |
|---|---|---|---|---|
| 0 | 131.9 | 88 | 148.9 | 1,586 |
| 1 | **164.7** | **121** | 159.1 | 1,657 |
| 2 | 142.4 | 105 | 137.6 | 1,575 |
| 3 | 153.4 | 109 | 216.3 | **5,411** |
| 4 | 149.4 | 103 | 161.5 | 1,567 |

**Key Observations:**
- **Class 1 (Depression)** posts are longest on average — individuals may write more extensively when processing depressive thoughts
- **Class 3** has the highest variance and the longest single post (5,411 words) — suggesting some very detailed personal accounts
- High standard deviations across all classes indicate highly variable post lengths — typical of free-form social media writing

### Title Length

| Statistic | Value |
|---|---|
| Mean | 43.9 characters |
| Median | 35 characters |
| Max | 300 characters |

---

## 7. Data Quality Issues Identified

| Issue | Count | Impact | Proposed Action |
|---|---|---|---|
| Missing `text` (null body) | 350 (5.9%) | Medium | Use `title` as fallback; or combine title+text |
| URLs in text | 344 posts | Low | Remove during preprocessing (Stage 3) |
| Reddit markup (`**`, `&amp;`, `/r/`) | 284 posts | Low | Strip HTML entities and markdown in Stage 3 |
| Duplicate titles | 1,306 | Medium | Investigate — may be pinned/repeated mod posts |
| Extremely long posts (>2,000 words) | ~50 posts | Low | May need truncation for BERT (max 512 tokens) |
| No duplicate rows | 0 | — | No action needed |

---

## 8. Sample Records per Class

### Class 0 — General Mental Health
- **Title:** "Free Covid-19 Anxiety e-Workbook. Please, take care of yourselves..."
- **Pattern:** Often informational posts, resource sharing, mod announcements

### Class 1 — Depression
- **Title:** "Regular check-in post, with information about our rules and wikis"
- **Pattern:** Personal struggles, self-reflection, first-person emotional accounts

### Class 2 — Anxiety
- **Title:** "Announcement on our AMA with Dr. Tracey Marks"
- **Pattern:** Mix of personal posts and community announcements

### Class 3 — Avoidant PD / Bipolar
- **Title:** "Is there anyone interested in joining a group for AvPD on Telegram?"
- **Pattern:** Community-seeking behavior, detailed personal narratives

### Class 4 — Suicidal Ideation / PTSD
- **Title:** "Set your intention"
- **Pattern:** Weekly/recurring threads, personal crises, coping strategies

---

## 9. Feature Strategy

Based on the data understanding, the following feature engineering decisions are proposed:

| Feature | Description | Rationale |
|---|---|---|
| `combined_text` | `title + " " + text` (with null handling) | Maximizes available signal |
| `text_only` | `text` field (NaN → empty string) | Pure body content |
| `title_only` | `title` field | Useful for 350 null-text cases |
| `word_count` | Number of words in combined text | May have class-level predictive power |
| `text_missing` | Binary flag: 1 if text was null | Could be informative for the model |

---

## 10. Data Collection Limitations

1. **Selection bias:** Reddit users are not representative of the general population (skewed toward younger, English-speaking, tech-literate users)
2. **Self-labeling:** Posts are labeled by subreddit, not by clinical diagnosis
3. **Comorbidity:** Users with depression may also post in anxiety subreddits — class overlap is plausible
4. **Temporal data not captured:** No timestamps available; seasonal/time-of-day patterns cannot be studied
5. **Anonymity:** No user-level features possible (nor ethically appropriate)

---

## 11. Key Takeaways for Next Stages

- ✅ Dataset is balanced — no resampling needed
- ⚠️ 5.9% null text — must use `title` as fallback
- ⚠️ Significant Reddit-specific noise (URLs, markdown, HTML entities) — aggressive preprocessing required
- ✅ No duplicate rows
- ⚠️ 1,306 duplicate titles — investigate mod/pinned posts during EDA
- ⚠️ BERT token limit (512) may truncate very long posts — evaluate impact

**Next Step → Stage 3: Preprocessing & Cleaning**

---
*Mental Health Status Classification Project | Stage 2 of 10*
