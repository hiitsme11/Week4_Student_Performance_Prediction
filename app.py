import os
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'student_score_model.joblib')
model = None

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    print("Warning: Model file not found. Please run train_model.py first.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', error="Model is not trained/loaded.")
    
    try:
        study_hours = float(request.form['study_hours'])
        attendance = float(request.form['attendance'])
        previous_score = float(request.form['previous_score'])
        assignments = int(request.form['assignments'])

        # Input validation
        if not (0 <= study_hours <= 24):
            return render_template('index.html', error="Study Hours must be between 0 and 24.")
        if not (0 <= attendance <= 100):
            return render_template('index.html', error="Attendance must be between 0% and 100%.")
        if not (0 <= previous_score <= 100):
            return render_template('index.html', error="Previous Score must be between 0 and 100.")
        if not (0 <= assignments <= 15):
            return render_template('index.html', error="Assignments must be between 0 and 15.")

        features = pd.DataFrame([[study_hours, attendance, previous_score, assignments]], 
                                columns=['Study_Hours', 'Attendance', 'Previous_Score', 'Assignments_Completed'])
        
        prediction = model.predict(features)[0]
        prediction = round(float(prediction), 2)
        
        result_status = "Pass" if prediction >= 50 else "Needs Improvement"
        status_class = "pass" if prediction >= 50 else "fail"

        return render_template('index.html', 
                               prediction=prediction, 
                               result_status=result_status,
                               status_class=status_class,
                               study_hours=study_hours,
                               attendance=attendance,
                               previous_score=previous_score,
                               assignments=assignments)

    except ValueError:
        return render_template('index.html', error="Invalid input. Please enter valid numerical values.")

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.get_json()
        study_hours = float(data.get('Study_Hours'))
        attendance = float(data.get('Attendance'))
        previous_score = float(data.get('Previous_Score'))
        assignments = int(data.get('Assignments_Completed'))

        features = pd.DataFrame([[study_hours, attendance, previous_score, assignments]], 
                                columns=['Study_Hours', 'Attendance', 'Previous_Score', 'Assignments_Completed'])
        
        prediction = float(model.predict(features)[0])
        prediction = round(prediction, 2)
        result_status = "Pass" if prediction >= 50 else "Needs Improvement"

        return jsonify({
            'status': 'success',
            'predicted_score': prediction,
            'predicted_result': result_status
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    model_loaded = model is not None
    return jsonify({
        'status': 'healthy' if model_loaded else 'unhealthy',
        'model_loaded': model_loaded
    }), 200 if model_loaded else 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
