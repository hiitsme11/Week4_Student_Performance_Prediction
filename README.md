# Student Performance Prediction Web Application

An end-to-end Machine Learning Web Application built with Flask and Scikit-learn that predicts student final academic performance based on study habits and past scores.

## Features
- **Machine Learning Pipeline**: Trained using Random Forest Regressor ($R^2 \approx 0.96$).
- **Interactive Web Interface**: Clean HTML5/CSS3 frontend to submit inputs and view instant predictions.
- **RESTful API Endpoint**: JSON endpoint `/api/predict` for external integrations.
- **Health Check Endpoint**: `/health` to verify model status.

## Project Structure
```
Week4_Student_Performance_Prediction/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── dataset/
│   └── student_performance.csv
├── models/
│   └── student_score_model.joblib
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── output/
│   ├── model_metrics.csv
│   └── test_predictions.csv
└── screenshots/
```

## Quick Start Guide

### 1. Create Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Run Training (Optional)
```powershell
python train_model.py
```

### 4. Run Flask Application
```powershell
python app.py
```
Open browser at `http://127.0.0.1:5000`
