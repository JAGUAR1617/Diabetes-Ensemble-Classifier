# app.py

import warnings
warnings.filterwarnings("ignore")

# --- Core libs
import numpy as np
import pandas as pd

# --- Streamlit/UI
import streamlit as st

# --- Plots
import matplotlib.pyplot as plt
import seaborn as sns

# --- ML: preprocessing / split / metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    roc_curve,
    classification_report,
    roc_auc_score,
)

# --- ML: models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, VotingClassifier
from sklearn import neighbors, linear_model, svm, naive_bayes

# --- XGBoost
from xgboost import XGBClassifier

# --- Keras/TensorFlow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import ModelCheckpoint

# --- Images (optional illustrations)
from PIL import Image


# =========================
# Streamlit page config & style
# =========================
st.set_page_config(page_title="Diabetes Ensemble Classifiers", layout="wide")

st.markdown(
    """
    <style>
      .stApp { background-color: #737378; }
      .block-container { padding-top: 1rem; padding-bottom: 2rem; }
      h1, h2, h3, h4, h5, h6, p { color: #fff; }
      .stAlert, .stDataFrame, .stTable { color: initial; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("""
# Diabetes dataset classification using ensemble learning  
**Author:** Panchanand Jha
""")

st.markdown("""
This Pima dataset consists of 8 features: pregnancies, plasma glucose, diastolic BP, skinfold thickness, 2-hour serum insulin, BMI, diabetes pedigree, and age.  
We'll train and compare multiple learners (Bagging, Boosting, Voting) and a Keras Sequential model.
""")


# =========================
# Load data (robust to stray spaces in headers)
# =========================
@st.cache_data
def load_data(path: str = "Diabeted_Ensemble.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()  # strip whitespace
    return df

try:
    data = load_data()
except FileNotFoundError:
    st.error("Could not find `diabetes.csv` in the working directory.")
    st.stop()

show_df = st.checkbox("Show dataset head")
if show_df:
    st.dataframe(data.head())

# Ensure the target column is named 'Outcome'
if "Outcome" not in data.columns:
    if "Class variable" in data.columns:
        data = data.rename(columns={"Class variable": "Outcome"})
    else:
        st.error("Target column not found. Expecting `Outcome` or `Class variable`.")
        st.stop()

# Encode Outcome if not numeric
if not pd.api.types.is_numeric_dtype(data["Outcome"]):
    data["Outcome"] = LabelEncoder().fit_transform(data["Outcome"])

# Features/Target
X = data.drop(columns=["Outcome"])
y = data["Outcome"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)


# =========================
# 1) Bagging Classifier
# =========================
st.header("1) Bagging Classifier (Decision Tree)")

bag_clf = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=200,
    bootstrap=True,
    n_jobs=-1,
    random_state=42,
)
bag_clf.fit(X_train_s, y_train)
bag_pred = bag_clf.predict(X_test_s)
bag_proba = bag_clf.predict_proba(X_test_s)[:, 1]
bag_acc = accuracy_score(y_test, bag_pred)


# =========================
# 2) XGBoost
# =========================
st.header("2) XGBoost Classifier")

xgb = XGBClassifier(
    learning_rate=0.05,
    n_estimators=300,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    gamma=0.0,
    booster="gbtree",
    random_state=27,
    n_jobs=-1,
)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
xgb_proba = xgb.predict_proba(X_test)[:, 1]
xgb_acc = accuracy_score(y_test, xgb_pred)


# =========================
# 3) Hard Voting
# =========================
st.header("3) Hard Voting (KNN + Perceptron + SVM)")

learner_1 = neighbors.KNeighborsClassifier(n_neighbors=5)
learner_2 = linear_model.Perceptron(tol=1e-3, random_state=42)
learner_3 = svm.SVC(gamma=0.001, probability=False, random_state=42)

hard_vote = VotingClassifier(
    estimators=[("KNN", learner_1), ("PRC", learner_2), ("SVM", learner_3)],
    voting="hard",
    n_jobs=-1,
)
hard_vote.fit(X_train_s, y_train)
hard_pred = hard_vote.predict(X_test_s)
hard_acc = accuracy_score(y_test, hard_pred)


# =========================
# 4) Soft Voting
# =========================
st.header("4) Soft Voting (KNN + NB + SVM)")

learner_4 = neighbors.KNeighborsClassifier(n_neighbors=5)
learner_5 = naive_bayes.GaussianNB()
learner_6 = svm.SVC(gamma=0.001, probability=True, random_state=42)

soft_vote = VotingClassifier(
    estimators=[("KNN", learner_4), ("NB", learner_5), ("SVM", learner_6)],
    voting="soft",
    n_jobs=-1,
)
soft_vote.fit(X_train_s, y_train)
soft_pred = soft_vote.predict(X_test_s)
soft_proba = soft_vote.predict_proba(X_test_s)[:, 1]
soft_acc = accuracy_score(y_test, soft_pred)


# =========================
# 5) Keras Sequential Model
# =========================
st.header("5) Keras Sequential Model")

X_train_nn = X_train_s
X_test_nn = X_test_s
y_train_nn = y_train.values
y_test_nn = y_test.values

model = Sequential([
    Input(shape=(X_train_nn.shape[1],)),   # ✅ fixed input
    Dense(32, activation="relu"),
    Dense(16, activation="relu"),
    Dense(8, activation="relu"),
    Dense(1, activation="sigmoid"),
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

ckpt_model = "pima-weights.best.hdf5"
checkpoint = ModelCheckpoint(
    filepath=ckpt_model,
    monitor="val_accuracy",   # ✅ fixed
    verbose=1,
    save_best_only=True,
    mode="max",
)

history = model.fit(
    X_train_nn, y_train_nn,
    epochs=50,
    batch_size=16,
    validation_data=(X_test_nn, y_test_nn),
    callbacks=[checkpoint],
    verbose=0,
)

nn_scores = model.evaluate(X_test_nn, y_test_nn, verbose=0)
nn_acc = float(nn_scores[1])

# =========================
# Comparison
# =========================
st.subheader("Model Accuracy Comparison")

comp = pd.DataFrame({
    "Model": ["Bagging", "XGBoost", "Hard Voting", "Soft Voting", "Keras NN"],
    "Accuracy": [bag_acc, xgb_acc, hard_acc, soft_acc, nn_acc],
})
st.dataframe(comp.style.format({"Accuracy": "{:.4f}"}), use_container_width=True)

fig = plt.figure()
plt.barh(comp["Model"], comp["Accuracy"])
plt.xlabel("Accuracy")
plt.title("Comparison of Models")
st.pyplot(fig)
