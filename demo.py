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
from tensorflow.keras.layers import Dense
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

The main objective is to compare several ensemble learning algorithms (Bagging, AdaBoost/Boosting/XGBoost, Hard/Soft Voting) and a simple Keras Sequential model on the Pima Indians Diabetes dataset.
""")

st.markdown("""
## Introduction
This Pima dataset consists of 8 features: pregnancies, plasma glucose, diastolic BP, skinfold thickness, 2-hour serum insulin, BMI, diabetes pedigree, and age.  
We'll train and compare multiple learners and visualize metrics along the way.
""")


# =========================
# Load data (robust to stray spaces in headers)
# =========================
@st.cache_data
def load_data(path: str = "Diabeted_Ensemble.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    # strip whitespace in column names to avoid KeyError when renaming
    df.columns = df.columns.str.strip()
    return df

try:
    data = load_data()
except FileNotFoundError:
    st.error("Could not find `Diabeted_Ensemble.csv` in the working directory.")
    st.stop()

show_df = st.checkbox("Show head of dataset")
if show_df:
    st.write(
        "Dataset preview (first 5 rows). If your target column isn’t recognized, ensure it’s named `Class variable` or `Outcome`."
    )
    st.dataframe(data.head())

# Ensure the target column is named 'Outcome'
if "Outcome" not in data.columns:
    # Original code expected a leading space; we've stripped headers, so rename cleanly
    if "Class variable" in data.columns:
        data = data.rename(columns={"Class variable": "Outcome"})
    else:
        st.error("Target column not found. Expecting a column named `Outcome` or `Class variable`.")
        st.stop()

# Encode Outcome if it's not numeric 0/1 already
if not pd.api.types.is_numeric_dtype(data["Outcome"]):
    data["Outcome"] = LabelEncoder().fit_transform(data["Outcome"])

# Features/Target
X = data.drop(columns=["Outcome"])
y = data["Outcome"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Scale features for (most) models and NN
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

labels_box = st.checkbox("Show label distribution (Outcome)")
if labels_box:
    st.write("Label counts:", y.value_counts())


# =========================
# 1) Bagging (Decision Tree base learner)
# =========================
st.header("1) Bagging Classifier (Decision Tree base learner)")

bag_clf = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=200,
    bootstrap=True,
    n_jobs=-1,
    random_state=42,
)
st.write(bag_clf)

bag_clf.fit(X_train_s, y_train)
bag_pred = bag_clf.predict(X_test_s)
bag_proba = bag_clf.predict_proba(X_test_s)[:, 1]
bag_cm = confusion_matrix(y_test, bag_pred)
bag_acc = accuracy_score(y_test, bag_pred)

cm1 = st.checkbox("Show confusion matrix & report for Bagging")
if cm1:
    st.write("Confusion matrix (Bagging)")
    fig = plt.figure()
    sns.heatmap(bag_cm, annot=True, fmt="d")
    st.pyplot(fig)
    st.text("Classification report (Bagging)")
    st.text(classification_report(y_test, bag_pred))
    st.write(f"Accuracy (Bagging): **{bag_acc*100:.2f}%**  |  ROC AUC: **{roc_auc_score(y_test, bag_proba):.3f}**")

roc1 = st.checkbox("Show ROC curve (Bagging)")
if roc1:
    fpr, tpr, _ = roc_curve(y_test, bag_proba)
    fig = plt.figure()
    plt.plot(fpr, tpr, label="Bagging")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Bagging")
    plt.legend()
    st.pyplot(fig)


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
    booster="gbtree",        # 'dart' can work but is slower; 'gbtree' is common
    random_state=27,
    n_jobs=-1,
)
st.write(xgb)

xgb.fit(X_train, y_train)  # XGB handles unscaled features well
xgb_pred = xgb.predict(X_test)
xgb_proba = xgb.predict_proba(X_test)[:, 1]
xgb_cm = confusion_matrix(y_test, xgb_pred)
xgb_acc = accuracy_score(y_test, xgb_pred)

cm2 = st.checkbox("Show confusion matrix & report for XGBoost")
if cm2:
    st.write("Confusion matrix (XGBoost)")
    fig2 = plt.figure()
    sns.heatmap(xgb_cm, annot=True, fmt="d")
    st.pyplot(fig2)
    st.text("Classification report (XGBoost)")
    st.text(classification_report(y_test, xgb_pred))
    st.write(f"Accuracy (XGBoost): **{xgb_acc*100:.2f}%**  |  ROC AUC: **{roc_auc_score(y_test, xgb_proba):.3f}**")

roc2 = st.checkbox("Compare ROC: Bagging vs XGBoost")
if roc2:
    fpr_b, tpr_b, _ = roc_curve(y_test, bag_proba)
    fpr_x, tpr_x, _ = roc_curve(y_test, xgb_proba)
    fig3 = plt.figure()
    plt.plot(fpr_b, tpr_b, label="Bagging")
    plt.plot(fpr_x, tpr_x, label="XGBoost")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Bagging vs XGBoost")
    plt.legend()
    st.pyplot(fig3)


# =========================
# 3) Hard Voting
# =========================
st.header("3) Hard Voting Classifier (KNN + Perceptron + SVM)")

learner_1 = neighbors.KNeighborsClassifier(n_neighbors=5)
learner_2 = linear_model.Perceptron(tol=1e-3, random_state=42)
learner_3 = svm.SVC(gamma=0.001, probability=False, random_state=42)

hard_vote = VotingClassifier(
    estimators=[("KNN", learner_1), ("PRC", learner_2), ("SVM", learner_3)],
    voting="hard",
    n_jobs=-1,
)
hard_vote.fit(X_train_s, y_train)  # use scaled for KNN/Perceptron/SVM
hard_pred = hard_vote.predict(X_test_s)
hard_cm = confusion_matrix(y_test, hard_pred)
hard_acc = accuracy_score(y_test, hard_pred)

cm3 = st.checkbox("Show confusion matrix & report for Hard Voting")
if cm3:
    st.write("Confusion matrix (Hard Voting)")
    fig4 = plt.figure()
    sns.heatmap(hard_cm, annot=True, fmt="d")
    st.pyplot(fig4)
    st.text("Classification report (Hard Voting)")
    st.text(classification_report(y_test, hard_pred))
    st.write(f"Accuracy (Hard Voting): **{hard_acc*100:.2f}%**")


# =========================
# 4) Soft Voting
# =========================
st.header("4) Soft Voting Classifier (KNN + GaussianNB + SVM-prob)")

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
soft_cm = confusion_matrix(y_test, soft_pred)
soft_acc = accuracy_score(y_test, soft_pred)

cm4 = st.checkbox("Show confusion matrix & report for Soft Voting")
if cm4:
    st.write("Confusion matrix (Soft Voting)")
    fig5 = plt.figure()
    sns.heatmap(soft_cm, annot=True, fmt="d")
    st.pyplot(fig5)
    st.text("Classification report (Soft Voting)")
    st.text(classification_report(y_test, soft_pred))
    # Base learner accuracies (optional)
    st.write(
        "Base learner accuracies:",
        {
            "KNN": accuracy_score(y_test, learner_4.fit(X_train_s, y_train).predict(X_test_s)),
            "NB": accuracy_score(y_test, learner_5.fit(X_train_s, y_train).predict(X_test_s)),
            "SVM": accuracy_score(y_test, learner_6.fit(X_train_s, y_train).predict(X_test_s)),
        },
    )
    st.write(f"Accuracy (Soft Voting): **{soft_acc*100:.2f}%**  |  ROC AUC: **{roc_auc_score(y_test, soft_proba):.3f}**")


# =========================
# Feature Importance from XGBoost
# =========================
st.subheader("Feature Importances (XGBoost)")
imp_feature = pd.DataFrame(
    {"Feature": X.columns.tolist(), "Importance": xgb.feature_importances_}
).sort_values("Importance", ascending=True)

fig6 = plt.figure(figsize=(8, 4))
plt.title("Feature Importance (XGBoost)")
plt.barh(imp_feature["Feature"], imp_feature["Importance"])
st.pyplot(fig6)
st.dataframe(imp_feature.sort_values("Importance", ascending=False), use_container_width=True)


# =========================
# 5) Keras Sequential Model (simple MLP)
# =========================
st.header("5) Keras Sequential Model")

# Use the same scaled train/test as classical models
X_train_nn = X_train_s
X_test_nn = X_test_s
y_train_nn = y_train.values if isinstance(y_train, pd.Series) else y_train
y_test_nn = y_test.values if isinstance(y_test, pd.Series) else y_test

model = Sequential([
    Dense(32, input_dim=X_train_nn.shape[1], activation="relu"),
    Dense(16, activation="relu"),
    Dense(8, activation="relu"),
    Dense(1, activation="sigmoid"),
])

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)

ckpt_model = "pima-weights.best.hdf5"
checkpoint = ModelCheckpoint(
    filepath=ckpt_model,
    monitor="val_accuracy",   # <-- FIXED from deprecated 'val_acc'
    verbose=1,
    save_best_only=True,
    mode="max",
)

st.write("Starting NN training...")
history = model.fit(
    X_train_nn, y_train_nn,
    epochs=100,
    batch_size=16,
    validation_data=(X_test_nn, y_test_nn),
    callbacks=[checkpoint],
    verbose=0,
)

# Accuracy plot
fig7 = plt.figure()
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model Accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "val"])
st.pyplot(fig7)

# Loss plot
fig8 = plt.figure()
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model Loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "val"])
st.pyplot(fig8)

nn_scores = model.evaluate(X_test_nn, y_test_nn, verbose=0)
nn_acc = float(nn_scores[1])
st.write(f"Neural Net Accuracy: **{nn_acc*100:.2f}%**")


# =========================
# Comparison
# =========================
st.markdown("## Comparison of Algorithms")
comp = pd.DataFrame({
    "model": ["Bagging", "XGBoost", "Hard Voting", "Soft Voting", "Keras Sequential"],
    "accuracy": [bag_acc, xgb_acc, hard_acc, soft_acc, nn_acc],
})
st.dataframe(comp.style.format({"accuracy": "{:.4f}"}), use_container_width=True)

fig9 = plt.figure()
plt.barh(comp["model"], comp["accuracy"])
plt.xlabel("Accuracy")
plt.title("Model Accuracy Comparison")
st.pyplot(fig9)

st.markdown(
    """
    *Tip to improve results further:* try cross-validation, hyperparameter tuning (e.g., tree depth, learning rate, number of estimators), and stronger neural nets. Ensure proper handling of missing values/outliers and keep features scaled for distance-based models and neural networks.
    """
)

