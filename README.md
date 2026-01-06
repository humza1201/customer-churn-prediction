# 🚀 Customer Churn Prediction Web App

An end-to-end **Data Science & Machine Learning project** that predicts customer churn using a production-grade ML pipeline and a Flask web application.

This project covers the complete lifecycle:
📊 EDA → ⚙️ Feature Engineering → 🤖 ML Modeling → 🌐 Deployment

---

## 🧠 Problem Statement
Customer churn is a major challenge for subscription-based businesses.  
The goal of this project is to **identify customers who are likely to churn**, allowing businesses to take proactive retention actions.

---

## 📊 Dataset
- **Source:** Telecom Customer Churn dataset (Kaggle)
- **Rows:** 7,043
- **Features:** 20 customer attributes + 1 target (`Churn`)
- **Target Variable:** `Churn` (Yes / No)

---

## 🔍 Exploratory Data Analysis (EDA)
Key insights discovered during EDA:
- 📉 **Low-tenure customers churn the most**
- 📄 **Month-to-month contracts have the highest churn**
- 💰 **Higher monthly charges → higher churn risk**
- 🌐 **Fiber optic users churn significantly more**
- 💳 **Electronic check payment method shows highest churn**

---

## ⚙️ Feature Engineering & Preprocessing
- Handled hidden missing values in `TotalCharges`
- Business-logic-based imputation
- Feature categorization:
  - 🔢 Numerical features → Scaled
  - 🔵 Binary categorical → Encoded
  - 🟣 Multi-category categorical → One-Hot Encoded
- Used `Pipeline` and `ColumnTransformer` to prevent data leakage

---

## 🤖 Machine Learning Model
- **Model:** Logistic Regression
- **Reason:** Interpretability, probability outputs, strong baseline for churn
- **Class Imbalance Handling:** `class_weight='balanced'`

### 📈 Model Performance
| Metric | Value |
|------|------|
| Accuracy | **74.82%** |
| Recall (Churn = Yes) | **0.78** |
| F1-score | **0.62** |
| ROC-AUC | **0.84** |

📌 **Recall was prioritized** to minimize missed churners (business-critical).

---

## 💼 Business Cost Framing
- Missing a churner (False Negative) is far more expensive than a false alarm
- Model optimized to **maximize churn capture**
- Suitable for real-world retention strategies

---

## 🌐 Web Application (Flask)
- User-friendly web interface
- Takes customer details via form
- Outputs:
  - 📊 Churn Probability
  - 🚦 Risk Level (Low / Medium / High)

---

## 🗂️ Project Structure

customer-churn-prediction/
│
├── app.py
├── train_logistic_regression.py
│
├── artifacts/
│   ├── model.pkl
│   └── pipeline.pkl
│
├── notebooks/
│   ├── eda.ipynb
│   ├── exploring_data.ipynb   
│   ├── main.ipynb
│   ├── train_random_forest.ipynb
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
├── requirements.txt
└── README.md



## ▶️ How to Run Locally
```bash
pip install -r requirements.txt
python app.py