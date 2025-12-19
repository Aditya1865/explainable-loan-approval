📄 README.md (DOWNLOADABLE CONTENT)
# 🏦 Explainable AI for Loan Approval

## 📌 Overview
This project implements a production-style Explainable AI system for predicting loan approval decisions.  
It combines a supervised machine learning model, a FastAPI backend, and a Streamlit frontend, with model explainability and fairness analysis using SHAP and LIME.

---

## 🚀 Key Features
- Loan approval prediction using supervised ML
- REST API built with FastAPI
- Interactive UI using Streamlit
- SHAP explanations for feature importance
- LIME explanations for individual predictions
- Fairness analysis across demographic groups
- Model evaluation using Accuracy and ROC-AUC

---

## 🧠 Tech Stack
- Python
- Scikit-learn
- FastAPI
- Streamlit
- SHAP
- LIME
- Pandas, NumPy
- Uvicorn

---

## 📂 Project Structure
explainable-loan-approval/
│
├── backend/
│ ├── main.py
│ └── model.pkl
│
├── frontend/
│ └── app.py
│
├── data/
│ └── loan_data.csv
│
├── train_model.py
├── evaluate_model.py
├── requirements.txt
└── README.md


---

## ⚙️ How to Run

### 1️⃣ Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```
2️⃣ Install Dependencies
```
pip install -r requirements.txt
```
3️⃣ Train the Model
```
python train_model.py
```

4️⃣ Start FastAPI Backend
```
uvicorn backend.main:app --reload
Open:
http://127.0.0.1:8000/docs
```

5️⃣ Start Streamlit Frontend
```
streamlit run frontend/app.py
```


🔍 Explainability
SHAP
Explains feature impact on predictions

Helps understand model behavior and bias

LIME
Explains individual loan decisions

Improves trust in automated decisions

⚖️ Fairness Analysis
Approval rates are compared across demographic groups (e.g., gender) to detect potential bias and ensure equitable decision-making.

📊 Model Evaluation
Evaluation is performed using a train-test split to avoid data leakage.

Metrics:

Accuracy

ROC-AUC

📌 Resume Description
Explainable AI for Loan Approval
Built a production-ready ML system using FastAPI and Streamlit to predict loan approvals.
Integrated SHAP and LIME for explainability, performed fairness analysis, and evaluated performance using Accuracy and ROC-AUC.

🔮 Future Improvements
Cloud deployment

Dockerization

Authentication

Advanced fairness metrics

Model monitoring

👨‍💻 Author
Aditya Pawar
