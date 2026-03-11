import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


st.title("Anomaly Detection Dashboard")

st.write("""
This application detects anomalies in financial transaction data
using the Isolation Forest algorithm.
""")

# Load dataset
data = pd.read_csv("Finance_data.csv")

st.subheader("Dataset Preview")
st.dataframe(data.head())

# Select numerical features
X = data.select_dtypes(include=np.number)

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train models (only once)
@st.cache_resource
def train_model(X_scaled):

    if_model = IsolationForest(contamination=0.05, random_state=42) 
    lof_model = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True)  # novelty=True so the model can detect anomalies in new                                                                                               unseen data instead of only the training data
    svm_model = OneClassSVM(kernel='rbf', gamma='auto')
    

    if_model.fit(X_scaled)
    lof_model.fit(X_scaled)
    svm_model.fit(X_scaled)
    

    return {
        "Isolation Forest": if_model,
        "Local Outlier Factor": lof_model,
        "One-class SVM": svm_model,
        
    }

models = train_model(X_scaled)

# Model selector
model_choice = st.sidebar.selectbox(
    "Select Machine Learning Model",
    list(models.keys())
)

model = models[model_choice]

# Make app more interactive
st.sidebar.subheader("Model Parameters")

contamination = st.sidebar.slider(
    "Contamination (expected anomaly %)",
    0.01,
    0.20,
    0.05
)

n_neighbors = st.sidebar.slider(
    "LOF Neighbors",
    5,
    50,
    20
)

gamma = st.sidebar.selectbox(
    "SVM Gamma",
    ["scale", "auto"]
)

# Predict anomalies (these algorithms don't map X->y but learn the structure of the data itself) then add Anomaly column to dataset
prediction = model.predict(X_scaled)
data["Anomaly"] = prediction

# anomaly score (distance from normal behavior)
scores = model.decision_function(X_scaled)
data["Anomaly Score"] = scores

st.subheader("Anomaly Scores")
st.dataframe(data[["Anomaly", "Anomaly Score"]].head())

# Explain anomaly results
st.subheader("Anomaly Detection Results")
st.write("""
- **1** = Normal data  
- **-1** = Anomaly
""")

# Visualization
st.subheader("Anomaly Visualization")

# Separate normal points and anomalies
normal = data[data["Anomaly"] == 1]
anomalies = data[data["Anomaly"] == -1]

fig, ax = plt.subplots()

# Normal points
ax.scatter(
    normal.iloc[:,0],     # selects first column of X; : -> all rows, 0 -> first column; this becomes the x-axis
    normal.iloc[:,1],     # all rows of the second column; this becomes the y-axis
    c="blue",             # control the color of each point
    alpha=0.5,            # controls transparency
    label="Normal"
)

# Anomalies
ax.scatter(
    anomalies.iloc[:,0],
    anomalies.iloc[:,1],
    c="red",
    marker="x",
    s=100,
    label="Anomaly"
)

ax.set_xlabel(X.columns[0])
ax.set_ylabel(X.columns[1])
ax.set_title("Detected Anomalies")
ax.legend()

st.pyplot(fig)

# Comparison of all models
comparison_results = []
for name, model in models.items():
    prediction = model.predict(X_scaled)
    anomaly_count = (prediction == -1).sum()
    comparison_results.append({"Model": name,
                                                    "Anomaly count": anomaly_count})

# Create comparison table and bar chart
comparison_df = pd.DataFrame(comparison_results)
st.dataframe(comparison_df)
st.bar_chart(comparison_df.set_index("Model"))



