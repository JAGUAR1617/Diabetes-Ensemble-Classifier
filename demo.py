import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint

# loading dataset using pandas
import pandas as pd
import numpy as np
import base64

# visualisation
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from matplotlib import pyplot

# EDA
from collections import Counter

# data preprocessing
from sklearn.preprocessing import StandardScaler

# data splitting
from sklearn.model_selection import train_test_split

# data modeling
from sklearn.metrics import (
    confusion_matrix, accuracy_score, roc_curve,
    classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# ensembling
from xgboost import XGBClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model, svm, neighbors, naive_bayes

from PIL import Image

# ------------------ Streamlit style ------------------
st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: #737378
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------ Intro ------------------
"""
# Diabetes dataset classification using ensemble learning
## Panchanand Jha

The main objective of this work is to provide better classification model and comparison of various ensemble learning algorithms.
"""

"""
# Introduction
This Pima dataset consists of eight features...
"""

# ------------------ MAIN ------------------
def main():
    st.title("1. Bagging Classifier with decision tree as a base learner")

    # importing diabetes dataset
    data = pd.read_csv("Diabeted_Ensemble.csv")

    data1 = st.checkbox('check the box to see DataFrame')
    if data1:
        st.write('diabetes dataset', data.head())

    # output is class variable and lets rename it as Outcome
    data.rename(columns={' Class variable': 'Outcome'}, inplace=True, errors='raise')

    lb = LabelEncoder()
    data['Outcome'] = lb.fit_transform(data['Outcome'])

    # features + target
    X = data.drop('Outcome', axis=1)
    y = data["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=0
    )

    # normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ---------------- BAGGING ----------------
    model1 = 'baggingclassifier'
    bag_clf = BaggingClassifier(
        base_estimator=DecisionTreeClassifier(),
        n_estimators=500, bootstrap=True, n_jobs=1, random_state=42
    )
    bag_clf.fit(X_train, y_train)
    bag_predicted = bag_clf.predict(X_test)
    bag_conf_matrix = confusion_matrix(y_test, bag_predicted)
    bag_acc_score = accuracy_score(y_test, bag_predicted)

    st.write("Accuracy of Bagging:", bag_acc_score * 100)

    # ---------------- XGBOOST ----------------
    st.header("2. XGBoost Classifier")
    xgb = XGBClassifier(
        learning_rate=0.01, n_estimators=25, max_depth=15, gamma=0.6,
        subsample=0.52, colsample_bytree=0.6, seed=27,
        reg_lambda=2, booster='dart', colsample_bylevel=0.6, colsample_bynode=0.5
    )
    xgb.fit(X_train, y_train)
    xgb_predicted = xgb.predict(X_test)
    xgb_conf_matrix = confusion_matrix(y_test, xgb_predicted)
    xgb_acc_score = accuracy_score(y_test, xgb_predicted)

    st.write("Accuracy of XGBoost:", xgb_acc_score * 100)

    # ---------------- HARD VOTING ----------------
    st.header("3. Hard Voting Classifier")
    learner_1 = neighbors.KNeighborsClassifier(n_neighbors=5)
    learner_2 = linear_model.Perceptron(tol=1e-2, random_state=0)
    learner_3 = svm.SVC(gamma=0.001)
    voting = VotingClassifier([('KNN', learner_1), ('PRC', learner_2), ('SVM', learner_3)])
    voting.fit(X_train, y_train)
    hard_predictions = voting.predict(X_test)
    hard_accuracy = accuracy_score(y_test, hard_predictions)

    st.write("Accuracy of Hard Voting:", hard_accuracy)

    # ---------------- SOFT VOTING ----------------
    st.header("4. Soft Voting Classifier")
    learner_4 = neighbors.KNeighborsClassifier(n_neighbors=5)
    learner_5 = naive_bayes.GaussianNB()
    learner_6 = svm.SVC(gamma=0.001, probability=True)
    voting = VotingClassifier(
        [('KNN', learner_4), ('NB', learner_5), ('SVM', learner_6)], voting='soft'
    )
    voting.fit(X_train, y_train)
    soft_predictions = voting.predict(X_test)
    soft_accuracy = accuracy_score(y_test, soft_predictions)

    st.write("Accuracy of Soft Voting:", soft_accuracy)

    # ---------------- KERAS SEQUENTIAL ----------------
    st.header("5. Keras Sequential Model")

    df_d = pd.read_csv("Diabeted_Ensemble.csv")
    df_d.rename(columns={' Class variable': 'Outcome'}, inplace=True, errors='raise')
    lb = LabelEncoder()
    df_d['Outcome'] = lb.fit_transform(df_d['Outcome'])

    features = list(df_d.columns.values)
    features.remove('Outcome')

    Xx = df_d[features].values
    yy = df_d['Outcome'].values

    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(
        Xx, yy, test_size=0.25, random_state=0
    )

    model = Sequential()
    model.add(Dense(32, input_dim=X_train_1.shape[1], activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # ✅ FIX: must use .keras file extension
    ckpt_model = "pima-weights.best.keras"
    checkpoint = ModelCheckpoint(
        filepath=ckpt_model,
        monitor="val_accuracy",   # ✅ updated from val_acc
        verbose=1,
        save_best_only=True,
        mode="max"
    )
    callbacks_list = [checkpoint]

    st.write('Starting training...')
    history = model.fit(
        X_train_1, y_train_1,
        epochs=100,
        validation_data=(X_test_1, y_test_1),
        batch_size=8,
        callbacks=callbacks_list,
        verbose=0
    )

    # Model accuracy
    fig7 = plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'])
    st.pyplot(fig7)

    # Model loss
    fig8 = plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'])
    st.pyplot(fig8)

    # Final evaluation
    scores = model.evaluate(X_test_1, y_test_1, verbose=0)
    seq_accuracy = scores[1]
    st.write("Sequential model accuracy: %.2f%%" % (seq_accuracy * 100))

    # ---------------- COMPARISON ----------------
    st.header("## Comparison of algorithms")
    comp = pd.DataFrame()
    comp["model"] = ["Bagging", "Boosting", "Hard Voting", "Soft Voting", "Sequential"]
    comp["accuracy"] = [bag_acc_score, xgb_acc_score, hard_accuracy, soft_accuracy, seq_accuracy]

    st.dataframe(comp)

    fig9 = plt.figure()
    plt.barh(comp['model'], comp['accuracy'])
    st.pyplot(fig9)


if __name__ == '__main__':
    main()
