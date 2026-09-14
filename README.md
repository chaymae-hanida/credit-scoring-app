# 💳 Credit Scoring App

End-to-end machine learning project that predicts a client's loan default risk, deployed as a **REST API (FastAPI)** with a **web interface (Streamlit)**.

## 🎯 Overview
🔗 **[Live Demo](https://chaymaehanida123-credit-scoring-app-frontendapp-mgdd9k.streamlit.app/)** — Try it now, no installation needed!
Built on top of a Random Forest classifier trained on the [Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset), this project goes beyond a Jupyter notebook: the model is exposed through a production-style API and a user-friendly web app, so a non-technical user can get a real-time credit decision.

**Model performance:**
| Metric | Score |
|---|---|
| Accuracy | 92.9% |
| Precision | 97.4% |
| Recall | 68.9% |
| ROC-AUC | 0.93 |

## 🏗️ Architecture

```
credit-scoring-app/
├── model/              # Trained model + preprocessing artifacts (.pkl)
├── backend/            # FastAPI REST API
│   └── main.py
├── frontend/           # Streamlit web interface
│   └── app.py
└── train_and_save.py   # Training script
```

## 🚀 Tech Stack

- **Machine Learning:** scikit-learn (Random Forest Classifier)
- **Backend:** FastAPI, Pydantic
- **Frontend:** Streamlit
- **Data processing:** pandas, numpy

## ⚙️ How to run locally

### 1. Clone the repo and install dependencies
```bash
git clone <your-repo-url>
cd credit-scoring-app
```

### 2. Run the API (optional — the Streamlit app can also call the model directly)
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```
API docs available at `http://127.0.0.1:8000/docs`

### 3. Run the web interface
```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

## 📊 Example API request

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{
  "person_age": 27,
  "person_income": 45000,
  "person_home_ownership": "RENT",
  "person_emp_length": 3.0,
  "loan_intent": "EDUCATION",
  "loan_grade": "B",
  "loan_amnt": 10000,
  "loan_int_rate": 11.5,
  "loan_percent_income": 0.22,
  "cb_person_default_on_file": "N",
  "cb_person_cred_hist_length": 4
}'
```

## 👩‍💻 Author

Chaymae Hanida — Master's student in Machine Learning Avancé et Intelligence Multimédia (MLAIM), USMBA Fès.
"# credit-scoring-app" 
