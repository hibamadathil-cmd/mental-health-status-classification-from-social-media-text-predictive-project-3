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
| 1 | Problem definition & literature review | Hiba |
| 2 | Data collection & understanding | Hiba |
| 3 | Preprocessing & cleaning | Hiba |
| 4 | Exploratory data analysis | Manasa |
| 5 | Feature engineering & selection | Manasa |
| 6 | BERT fine-tuning + TF-IDF/SVM baseline | Manasa |
| 7 | Evaluation: sensitivity & specificity | Aleena |
| 8 | Interpretation & explainability | Aleena |
| 9 | Streamlit deployment | Aleena |
| 10 | Docs & GitHub | Aleena |

## 📊 Model Results
| Model | Accuracy | F1 (weighted) |
|-------|----------|---------------|
| TF-IDF + SVM | 81.3% | 81.4% |
| BERT (fine-tuned) | See Manasa's notebook | See Manasa's notebook |

## 🔍 Explainability 
- **LIME**: Word-level explanations for individual predictions per class
- **TF-IDF weights**: Global most important words per class from SVM coefficients
- **Word clouds**: Visual summary of class-specific vocabulary

## 🚀 Run the Streamlit App
```bash
pip install streamlit scikit-learn pandas numpy
streamlit run app.py
```

## 📁 Repository Structure
```
├── cleaned_dataset.csv                    # Preprocessed dataset (Hiba)
├── Dataset.zip                            # Raw dataset
├── Stage1_Problem_Definition_and_Literature_Review.md  (Hiba)
├── Stage2_Data_Collection_and_Understanding.md         (Hiba)
├── Stage3_Preprocessing_and_Cleaning.md                (Hiba)
├── Member_2_—_Model_Development_&_Evaluation.ipynb     (Manasa)
├── mental_health_classifier.ipynb                      (Manasa)
├── Stage8_Explainability.ipynb            # LIME + word clouds  (Aleena)
├── Stage9_Deployment.md                   # Deployment guide    (Aleena)
├── Stage10_Summary.md                     # Project summary     (Aleena)
├── app.py                                 # Streamlit app       (Aleena)
└── README.md                              # This file           (Aleena)
```

## 🛠️ Tech Stack
- Python, scikit-learn, HuggingFace Transformers
- BERT (bert-base-uncased), TF-IDF + LinearSVC
- LIME for explainability
- Streamlit for deployment
-
##app link
- https://6yfyvnqza8mhswzv3jy97t.streamlit.app/

## 👥 Team
| Name | Stages | Role |
|------|--------|------|
| Hiba | Stages 1–3 | Data collection, preprocessing & problem definition |
| Manasa | Stages 4–6 | EDA, feature engineering & model development |
| Aleena | Stages 7–10 | Evaluation, explainability, deployment & docs |


## 🌐 Live Demo
> The app was deployed and tested via Streamlit + localtunnel on Google Colab.
> To run it yourself:
> ```bash
> pip install streamlit scikit-learn pandas numpy
> streamlit run app.py
> ```
> The model (`svm_model.pkl`) and label encoder (`le.pkl`) are included in this repo.
> On first run without the pkl files, the app trains the SVM automatically from `cleaned_dataset.csv`.
