Project Overview

This project builds an interactive machine learning dashboard for 
detecting anomalies in financial transaction data.
The application allows users to explore different anomaly detection 
algorithms and visualize unusual patterns directly through an 
interactive web interface.
The dashboard is built using Streamlit, enabling real-time model 
selection and visualization.


Live Application
 
https://anomaly-detection-4hvmcc9cuwbrbtzmfpws3d.streamlit.app

Features

Interactive anomaly detection dashboard
Multiple machine learning algorithms
Model comparison across algorithms
Anomaly scoring for each observation
Visual highlighting of anomalies
Adjustable model hyperparameters

Algorithms Implemented

The dashboard allows users to choose between several anomaly detection models:

- ️Isolation Forest
Uses random partitioning of data to isolate anomalies.
Isolation Forest is efficient for large datasets and works well with 
high-dimensional data.

-️ Local Outlier Factor
Detects anomalies by comparing the local density of a point with its neighbors.
Local Outlier Factor is effective for identifying local outliers within clusters.

- One-Class SVM
Learns a boundary around normal data points and flags points outside the boundary as 
anomalies.
One-Class SVM works well for complex non-linear data patterns.

Tech Stack

Python
Pandas
NumPy
Scikit-Learn
Matplotlib
Streamlit

Dashboard Capabilities

The application allows users to:
- Preview the dataset
- Select an anomaly detection algorithm
- View anomaly predictions
- Compare models by anomaly count
- Visualize anomalies in scatter plots
- Tune model parameters from the sidebarRun Locally

Clone the repository: git clone 

https://github.com/richygamo-ml/Anomaly-Detection.git

Install dependencies

pip install -r requirements.txt

Run the Streamlit app

https://anomaly-detection-4hvmcc9cuwbrbtzmfpws3d.streamlit.app

Model Comparison 

The dashboard compares the anomaly counts detected by each algorithm to help users 
evaluate model behavior.

Future Improvements

Possible enhancements include:
Adding more anomaly detection algorithms
Deploying the dashboard publicly
Adding interactive data uploads
Improving anomaly visualization

Author: Richy Gamo
Machine Learning & AI Projects

 Why This Project Matters
Anomaly detection is widely used in:
fraud detection
cybersecurity
healthcare monitoring
financial risk analysis

This project demonstrates how multiple anomaly detection algorithms can be combined 
into a user-friendly machine learning dashboard.
