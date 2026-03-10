import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

st.title("Anomaly Detection Dashboard")

st.write("""
This application detects anomalies in financial transaction data
using the Isolation Forest algorithm.
""")

# Load dataset
data = pd.read_csv("Finance_data.csv")

st.subheader("Dataset Preview")
st.dataframe(data.head())

# Select numerical features only
X = data.select_dtypes(include=np.number)

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train anomaly detection model
@st.cache_resource
def train_model(X_scaled):

    model = IsolationForest(
        contamination=0.05,
        random_state=42
    )

    model.fit(X_scaled)
    return model

model = train_model(X_scaled)

# Predict anomalies
anomaly_scores = model.predict(X_scaled)

data["Anomaly"] = anomaly_scores

st.subheader("Anomaly Detection Results")

st.write("""
- **1** = Normal data  
- **-1** = Anomaly
""")

st.dataframe(data.head())

# Visualization
st.subheader("Anomaly Visualization")

fig, ax = plt.subplots()

ax.scatter(
    X.iloc[:,0],
    X.iloc[:,1],
    c=data["Anomaly"],
    cmap="coolwarm",
    alpha=0.6
)

ax.set_xlabel(X.columns[0])
ax.set_ylabel(X.columns[1])
ax.set_title("Detected Anomalies")

st.pyplot(fig)

# Anomaly count
anomaly_count = (data["Anomaly"] == -1).sum()

st.metric("Number of Anomalies Detected", anomaly_count)