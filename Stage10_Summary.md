# Stage 10 — Project Summary & Findings
**Contributor: Member 3 — Aleena (Aleena11062004)**

## Project Goal
Classify social media text into 5 mental health categories using NLP and machine learning.

## Dataset
- Source: Reddit mental health subreddits
- Size: ~5,500 posts after cleaning
- Classes: Stress, Depression, Bipolar, Personality Disorder, Anxiety

## Models Built
| Model | Accuracy | F1 Score |
|-------|----------|----------|
| TF-IDF + LinearSVC | 81.3% | 81.4% |
| BERT (fine-tuned) | See Member 2 notebook | — |

## Key Findings from Explainability (Stage 8)
- **Stress (Class 0)**: Top words — stress, stressful, meditation, study
- **Depression (Class 1)**: Top words — depression, depressed, suicide, die
- **Bipolar (Class 2)**: Top words — bipolar, manic, episode, lithium
- **Personality Disorder (Class 3)**: Top words — avpd, social, avoidance, people
- **Anxiety (Class 4)**: Top words — anxiety, panic, ocd, heart, chest

## Member 3 Contributions (Aleena)
| Stage | Work Done |
|-------|-----------|
| 7 | Model evaluation: sensitivity, specificity, ROC/AUC, BERT vs SVM comparison |
| 8 | LIME word-level explanations, TF-IDF weight charts, word clouds per class |
| 9 | Streamlit web app — real-time mental health text classification |
| 10 | README documentation, project summary, GitHub contribution |

## Files Added by Member 3
- `app.py` — Streamlit application
- `Stage8_Explainability.ipynb` — Full explainability analysis notebook
- `Stage9_Deployment.md` — Deployment guide
- `Stage10_Summary.md` — This file
- `README.md` — Full project README
