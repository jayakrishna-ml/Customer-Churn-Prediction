# Customer Churn Prediction – Project

## Overview
This project analyzes telecom customer data to understand **customer churn behavior**
and builds machine learning models to **predict the probability of churn**.
It follows a complete **Data Science workflow**, including data preprocessing,
exploratory data analysis (EDA), feature engineering, model training,
evaluation, and deployment using a Streamlit web application.

---

## Objective
- Analyze customer behavior and churn patterns
- Identify key factors influencing customer churn
- Build and compare multiple classification models
- Deploy the best-performing model for real-time prediction

---

## Dataset
- **Source:** Telecom Customer Churn Dataset
- **Type:** Structured tabular dataset
- **Target Variable:** Churn (Yes / No)
- **Features Include:**
  - Customer demographics (gender, senior citizen)
  - Account information (tenure, contract type, payment method)
  - Service usage (internet service, streaming, tech support)
  - Charges (monthly charges, total charges)

---

## Tools & Technologies
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook

---

## Project Workflow
1. Data loading and understanding
2. Data cleaning and preprocessing
3. Feature engineering
   - Service count (`num_services`)
   - Total charges calculation
4. Exploratory Data Analysis (EDA)
5. Handling class imbalance
6. Model training using pipelines
7. Model evaluation and comparison
8. Best model selection
9. Model deployment using Streamlit

---

## Exploratory Data Analysis (EDA)
Key insights derived from EDA:
- Customers with **short tenure** have significantly higher churn
- **Month-to-month contracts** show the highest churn rate
- Higher **monthly charges** increase churn probability
- Customers with fewer subscribed services are more likely to churn

EDA insights directly influenced feature engineering and model selection.

---

## Machine Learning Models
- Logistic Regression
- Random Forest Classifier
- Support Vector Classifier
- XGBoost Classifier

All models were trained using consistent preprocessing pipelines.

---

## Model Evaluation
- **Metrics Used:**
  - ROC-AUC Score
  - Accuracy
  - Precision
  - Recall
- Best model selected based on **ROC-AUC and generalization performance**

---

## Deployment
- Interactive **Streamlit web application**
- User inputs customer details
- Application outputs:
  - Churn probability
  - Final churn decision (Likely / Unlikely)

---

## Repository Structure
```
Customer-Churn-Prediction/
├── app.py
├── eda_churn_analysis.ipynb
├── model_training.ipynb
├── best_churn_pipeline.pkl
├── categorical_values.pkl
├── requirements.txt
└── README.md
```

---

## How to Run
1. Clone or download the repository
2. Install dependencies:
    ```bash
     pip install -r requirements.txt
    ```
3. Run the Streamlit application
    ```bash
    streamlit run app.py
    ```
4. Open the app in your browser
   ```arduino
   http://localhost:8501
   ```

---

## Author
JK11
