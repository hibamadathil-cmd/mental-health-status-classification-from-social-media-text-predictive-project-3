# Stage 1: Problem Definition & Literature Review
## Mental Health Status Classification — Project Pipeline

---

## 1. Problem Statement

Mental health disorders, particularly depression, anxiety, and related conditions, represent one of the most significant public health challenges globally. Traditional clinical screening methods (e.g., PHQ-9, GAD-7) are limited by accessibility, stigma, and the availability of trained practitioners. With the explosive growth of social media platforms like Reddit, individuals increasingly express mental health struggles in online communities — creating a rich, real-world corpus of mental health language.

**Core Problem:**
> Can we build a reliable automated system that classifies the mental health status of Reddit posts into distinct categories using Natural Language Processing (NLP) and machine learning?

**Why it matters:**
- Early detection of mental health conditions can enable timely intervention
- Automated screening tools can support (not replace) clinical decision-making
- Large-scale analysis of mental health discourse can inform public health policy

---

## 2. Project Objectives

| # | Objective |
|---|-----------|
| 1 | Define a taxonomy of mental health status classes from Reddit data |
| 2 | Build and evaluate a baseline NLP classifier (TF-IDF + SVM) |
| 3 | Fine-tune a transformer-based model (BERT) for improved accuracy |
| 4 | Evaluate models using clinically meaningful metrics (sensitivity & specificity) |
| 5 | Deploy an interpretable, user-facing classification tool via Streamlit |

---

## 3. Dataset Overview (Preliminary)

- **Source:** Reddit mental health subreddits (r/depression, r/anxiety, r/SuicideWatch, r/mentalhealth, r/bipolar, etc.)
- **Format:** CSV with columns — `text` (post body), `title` (post title), `target` (class label)
- **Size:** 5,957 records
- **Target Classes:** 5 classes (labels 0–4), approximately balanced (~1,181–1,202 samples each)
- **Missing Values:** 350 records have missing `text` (body) — only title available

**Target Class Mapping (assumed based on subreddit origin):**

| Label | Presumed Class |
|-------|----------------|
| 0 | No mental health concern (control) |
| 1 | Depression |
| 2 | Anxiety |
| 3 | Bipolar Disorder |
| 4 | PTSD / Suicidal Ideation |

> Note: Exact label mapping to be confirmed during EDA (Stage 4).

---

## 4. Literature Review

### 4.1 NLP for Mental Health Detection

**Key foundational works:**

- **Gkotsis et al. (2016)** — One of the first studies to classify mental health conditions from Reddit posts using linguistic features. Demonstrated that subreddit-specific language patterns are highly predictive of mental health status.

- **Coppersmith et al. (2014, 2015)** — Pioneered the use of Twitter data for mental health classification. Introduced language model perplexity as a feature, highlighting statistical differences in language use among individuals with PTSD, depression, and bipolar disorder.

- **Yates et al. (2017)** — Used convolutional neural networks (CNNs) on Reddit posts for depression detection, outperforming traditional bag-of-words approaches.

- **Tadesse et al. (2019)** — Combined LIWC (Linguistic Inquiry and Word Count) features with machine learning classifiers. Found that emotional tone, cognitive processing language, and first-person pronoun use are strong predictors of depression.

### 4.2 Transformer Models in Mental Health NLP

- **Devlin et al. (2019) — BERT:** Bidirectional Encoder Representations from Transformers became the state-of-the-art for text classification tasks. Pre-trained on large corpora and fine-tunable for domain-specific tasks.

- **Ji et al. (2021)** — Systematically reviewed deep learning methods for mental health text analysis. Found BERT-based models consistently outperform traditional ML approaches, especially with limited data.

- **Mental-BERT / Mental-RoBERTa (Ji et al., 2022):** Domain-adapted BERT variants pre-trained on mental health corpora. These show significant gains over generic BERT for suicide ideation and depression detection.

- **Pérez et al. (2023):** Used LLMs with prompt engineering for zero-shot mental health classification, showing promise for low-resource settings.

### 4.3 Evaluation Metrics in Clinical NLP

Standard accuracy is insufficient in healthcare contexts. The literature emphasizes:

- **Sensitivity (Recall):** Proportion of true positive cases correctly identified — critical for not missing at-risk individuals.
- **Specificity:** Proportion of true negatives correctly identified — critical for avoiding false alarms.
- **AUC-ROC:** Area under the receiver-operating curve, evaluating overall discriminative power.
- **F1-Score (Macro):** Balances precision and recall across classes, especially important for near-balanced datasets.

> **Key insight from literature:** In mental health screening, **high sensitivity is prioritized** over high specificity, as missing a true case (false negative) carries greater clinical risk than a false positive.

### 4.4 Ethical Considerations

- **Stigma and labeling:** Automated labels can reinforce stigma. Models must not be used for definitive diagnosis.
- **Informed consent:** Reddit data is public, but users may not expect clinical analysis. IRB-equivalent ethical considerations apply.
- **Bias:** Models trained on majority-English, Reddit-using populations may not generalize to diverse populations.
- **Explainability:** Clinical stakeholders require interpretable models (LIME/SHAP), not black-box outputs.

---

## 5. Research Gap & Motivation

Despite substantial progress, several gaps motivate this project:

1. **Multi-class classification:** Most studies focus on binary (depressed vs. not) rather than multi-class settings across multiple disorders.
2. **Baseline comparison:** Few studies directly compare transformer models with classical TF-IDF+SVM baselines on the same dataset.
3. **Interpretability:** Limited work on explaining *which words/phrases* drive classification decisions.
4. **Deployment:** Most research prototypes are not deployed as accessible tools — this project bridges that gap with a Streamlit app.

---

## 6. Project Scope & Limitations

**In scope:**
- Binary and multi-class mental health status classification
- Text-only features (post body + title)
- English-language posts
- Comparison of classical ML vs. transformer-based approaches

**Out of scope:**
- Clinical validation or deployment in real healthcare settings
- User-level longitudinal analysis (only individual posts)
- Image/video data
- Non-English languages

---

## 7. Tools & Technologies (Planned)

| Category | Tools |
|----------|-------|
| Language | Python 3.10+ |
| Data Handling | pandas, numpy |
| NLP Preprocessing | NLTK, spaCy, regex |
| Classical ML | scikit-learn (TF-IDF, SVM, Logistic Regression) |
| Deep Learning | PyTorch, Hugging Face Transformers (BERT) |
| Visualization | matplotlib, seaborn, wordcloud |
| Explainability | SHAP, LIME |
| Deployment | Streamlit |
| Version Control | GitHub |

---

## 8. Summary

Stage 1 establishes the problem as a **multi-class text classification task** using Reddit mental health data, grounded in a robust body of NLP and clinical literature. The project adopts a dual-model strategy (TF-IDF/SVM baseline + BERT fine-tuning), prioritizes clinically meaningful evaluation, and plans for interpretability and open deployment.

**Next Step → Stage 2: Data Collection & Understanding**

---
*Mental Health Status Classification Project | Stage 1 of 10*
