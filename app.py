import streamlit as st
import pandas as pd, numpy as np, pickle, warnings
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Mental Health Classifier",page_icon="🧠",layout="centered")
CLASS_LABELS={0:("😰 Stress","#3498db"),1:("😞 Depression","#e74c3c"),
              2:("🔄 Bipolar Disorder","#9b59b6"),3:("🙈 Personality Disorder","#e67e22"),
              4:("😨 Anxiety Disorder","#1abc9c")}
MODEL_PATH="svm_model.pkl"; LE_PATH="le.pkl"; DATA_PATH="cleaned_dataset.csv"
@st.cache_resource
def load_model():
    if Path(MODEL_PATH).exists() and Path(LE_PATH).exists():
        with open(MODEL_PATH,"rb") as f: model=pickle.load(f)
        with open(LE_PATH,"rb") as f: le=pickle.load(f)
        return model,le
    df=pd.read_csv(DATA_PATH).dropna(subset=["clean_text","target"]).reset_index(drop=True)
    df["clean_text"]=df["clean_text"].astype(str)
    le=LabelEncoder(); texts=df["clean_text"].tolist()
    labels=le.fit_transform(df["target"]).tolist()
    X_train,_,y_train,_=train_test_split(texts,labels,test_size=0.10,random_state=42,stratify=labels)
    model=Pipeline([("tfidf",TfidfVectorizer(max_features=100000,ngram_range=(1,2),sublinear_tf=True,min_df=2,max_df=0.95)),
                    ("clf",CalibratedClassifierCV(LinearSVC(max_iter=2000,class_weight="balanced",random_state=42),cv=3,method="sigmoid"))])
    model.fit(X_train,y_train)
    with open(MODEL_PATH,"wb") as f: pickle.dump(model,f)
    with open(LE_PATH,"wb") as f: pickle.dump(le,f)
    return model,le
st.title("🧠 Mental Health Status Classifier")
st.markdown("Type or paste any text and the model will predict the mental health category.")
st.markdown("---")
with st.spinner("Loading model..."): model,le=load_model()
st.success("Model ready!")
user_input=st.text_area("Enter your text here:",height=180,placeholder="e.g. I feel really anxious lately...")
if st.button("🔍 Predict",use_container_width=True):
    if not user_input.strip(): st.warning("Please enter some text first.")
    else:
        probs=model.predict_proba([user_input])[0]; pred_idx=int(np.argmax(probs))
        pred_class=int(le.classes_[pred_idx]); label,color=CLASS_LABELS[pred_class]
        st.markdown("---")
        st.markdown(f"### Prediction: {label}",unsafe_allow_html=True)
        st.markdown(f"**Confidence: {probs[pred_idx]*100:.1f}%**")
        st.markdown("---"); st.markdown("#### Confidence scores for all classes:")
        for i,prob in enumerate(probs):
            cls=int(le.classes_[i]); lbl,clr=CLASS_LABELS[cls]
            st.markdown(f'''

              {lbl}
              
              {prob*100:.1f}%
''',unsafe_allow_html=True)
        st.caption("Model: TF-IDF + LinearSVC  |  Dataset: Reddit mental health posts")
