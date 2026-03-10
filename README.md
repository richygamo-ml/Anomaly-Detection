Anomaly Detection Dashboard

This project is an interactive machine learning dashboard that detects 
anomalies in financial transaction data using an unsupervised learning 
algorithm.
The application is built using Python and deployed as a web application 
with Streamlit.

Live Application
 
https://anomaly-detection-4hvmcc9cuwbrbtzmfpws3d.streamlit.app

Project Overview

The goal of this project is to identify unusual patterns in financial data 
that may indicate fraudulent or abnormal transactions.
The system uses an unsupervised anomaly detection algorithm to detect 
outliers in the dataset.
Users can explore the dataset, visualize anomalies, and see how the model 
identifies abnormal behavior.

Machine Learning Model

The project uses: Isolation Forest

Isolation Forest is an unsupervised learning algorithm designed to 
identify anomalies by isolating rare observations in the data.

Key characteristics:

• Works well with high-dimensional data
• Efficient for large datasets
• Commonly used for fraud detection and outlier detection

Tech Stack

Python
Pandas
NumPy
Scikit-Learn
Matplotlib
Streamlit

Dataset

The dataset contains financial transaction information used to identify 
abnormal behavior patterns.
The model analyzes numerical features in the dataset and identifies 
observations that deviate significantly from normal patterns.

Run Locally

Clone the repository: git clone 

https://github.com/richygamo-ml/Anomaly-Detection.git

Navigate to the project folder

cd Anomaly-Detection

Install dependencies

pip install -r requirements.txt

Run the Streamlit app

streamlit run Anomaly_Detection_App.py

Project Type

Machine Learning:
Unsupervised Learning
Anomaly Detection Dashboard
