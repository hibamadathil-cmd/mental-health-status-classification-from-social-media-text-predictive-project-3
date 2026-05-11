# 🧠 Mental Health Status Classification from Social Media Text

A machine learning pipeline that classifies social media posts into 5 mental health categories using NLP.

## 📋 Classes
| Label | Class |
|-------|-------|
| 0 | 😰 Stress |
| 1 | 😞 Depression |
| 2 | 🔄 Bipolar Disorder |
| 3 | 🙈 Personality Disorder |
| 4 | 😨 Anxiety Disorder |

## 🗂️ Project Pipeline
| Stage | Description | Member |
|-------|-------------|--------|
| 1 | Problem definition & literature review | Member 1 |
| 2 | Data collection & understanding | Member 1 |
| 3 | Preprocessing & cleaning | Member 1 |
| 4 | Exploratory data analysis | Member 2 |
| 5 | Feature engineering & selection | Member 2 |
| 6 | BERT fine-tuning + TF-IDF/SVM baseline | Member 2 |
| 7 | Evaluation: sensitivity & specificity | **Member 3 (Aleena)** |
| 8 | Interpretation & explainability | **Member 3 (Aleena)** |
| 9 | Streamlit deployment | **Member 3 (Aleena)** |
| 10 | Docs & GitHub | **Member 3 (Aleena)** |

## 📊 Model Results
| Model | Accuracy | F1 (weighted) |
|-------|----------|---------------|
| TF-IDF + SVM | 81.3% | 81.4% |
| BERT (fine-tuned) | See Member 2 notebook | See Member 2 notebook |

## 🔍 Explainability (Stage 8 — Member 3)
- **LIME**: Word-level explanations for individual predictions per class
- **TF-IDF weights**: Global most important words per class from SVM coefficients
- **Word clouds**: Visual summary of class-specific vocabulary

## 🚀 Run the Streamlit App (Stage 9 — Member 3)
```bash
pip install streamlit scikit-learn pandas numpy
streamlit run app.py
```

## 📁 Repository Structure
```
├── cleaned_dataset.csv                    # Preprocessed dataset (Member 1)
├── Dataset.zip                            # Raw dataset
├── Stage1_Problem_Definition_and_Literature_Review.md  (Member 1)
├── Stage2_Data_Collection_and_Understanding.md         (Member 1)
├── Stage3_Preprocessing_and_Cleaning.md                (Member 1)
├── Member_2_—_Model_Development_&_Evaluation.ipynb     (Member 2)
├── mental_health_classifier.ipynb                      (Member 2)
├── Stage8_Explainability.ipynb            # LIME + word clouds (Member 3)
├── Stage9_Deployment.md                   # Deployment guide   (Member 3)
├── Stage10_Summary.md                     # Project summary    (Member 3)
├── app.py                                 # Streamlit app      (Member 3)
└── README.md                              # This file          (Member 3)
```

## 🛠️ Tech Stack
- Python, scikit-learn, HuggingFace Transformers
- BERT (bert-base-uncased), TF-IDF + LinearSVC
- LIME for explainability
- Streamlit for deployment

## 👥 Team
| Member | Stages | GitHub |
|--------|--------|--------|
| Member 1 | Stages 1–3 | — |
| Member 2 | Stages 4–6 | — |
| **Member 3** | **Stages 7–10** | **Aleena11062004** |
