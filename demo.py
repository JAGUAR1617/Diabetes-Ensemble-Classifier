import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# ---------------------------
# Load dataset
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("diabetes.csv")  # change to your dataset path
    return df

df = load_data()
st.write("### Dataset Preview", df.head())

# ---------------------------
# Split features and target
# ---------------------------
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------
# Scale features
# ---------------------------
scaler = StandardScaler()
X_train_nn = scaler.fit_transform(X_train)
X_test_nn = scaler.transform(X_test)

# Convert labels if pandas Series
y_train_nn = y_train.values if isinstance(y_train, pd.Series) else y_train
y_test_nn = y_test.values if isinstance(y_test, pd.Series) else y_test

# ---------------------------
# Build Sequential model
# ---------------------------
model = Sequential([
    Input(shape=(X_train_nn.shape[1],)),  # ✅ Explicit Input layer
    Dense(32, activation="relu"),
    Dense(16, activation="relu"),
    Dense(8, activation="relu"),
    Dense(1, activation="sigmoid")        # Binary classification
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# ---------------------------
# Train model
# ---------------------------
history = model.fit(
    X_train_nn, y_train_nn,
    validation_data=(X_test_nn, y_test_nn),
    epochs=50,
    batch_size=16,
    verbose=0
)

# ---------------------------
# Evaluate
# ---------------------------
y_pred_prob = model.predict(X_test_nn)
y_pred = (y_pred_prob > 0.5).astype(int)

acc = accuracy_score(y_test_nn, y_pred)
cm = confusion_matrix(y_test_nn, y_pred)
report = classification_report(y_test_nn, y_pred, output_dict=True)

# ---------------------------
# Streamlit UI
# ---------------------------
st.write("### Model Performance")
st.write(f"**Accuracy:** {acc:.4f}")

st.write("#### Confusion Matrix")
st.write(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]))

st.write("#### Classification Report")
st.json(report)
