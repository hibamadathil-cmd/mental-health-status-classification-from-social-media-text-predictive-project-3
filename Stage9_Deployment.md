# Stage 9 — Streamlit Deployment

## Overview
The trained SVM model is deployed as an interactive web app using Streamlit.

## How to Run Locally
```bash
pip install streamlit scikit-learn pandas numpy
streamlit run app.py
```

## How to Run on Google Colab
```python
!pip install streamlit pyngrok --quiet
!npm install -g localtunnel
# run streamlit in background then expose via localtunnel
!lt --port 8501
```

## App Features
- Text input box for user to enter any social media post
- Predicts one of 5 mental health classes
- Shows confidence score for predicted class
- Displays confidence bars for all 5 classes

## Class Labels
| Class | Label |
|-------|-------|
| 0 | 😰 Stress |
| 1 | 😞 Depression |
| 2 | 🔄 Bipolar Disorder |
| 3 | 🙈 Personality Disorder |
| 4 | 😨 Anxiety Disorder |

## Model Used
TF-IDF (unigrams + bigrams) + LinearSVC with CalibratedClassifierCV
- Accuracy: 81.3%
- F1 (weighted): 81.4%
