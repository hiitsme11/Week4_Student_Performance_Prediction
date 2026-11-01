import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train():
    print("Starting Model Training Pipeline...")
    
    # Path setup
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, 'dataset', 'student_performance.csv')
    model_dir = os.path.join(base_dir, 'models')
    output_dir = os.path.join(base_dir, 'output')
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    df = pd.read_csv(data_path)
    print(f"Dataset loaded. Shape: {df.shape}")
    
    # Features & Target
    X = df[['Study_Hours', 'Attendance', 'Previous_Score', 'Assignments_Completed']]
    y = df['Final_Score']
    
    # Train-Test Split (80:20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model trained successfully.")
    
    # Predict & Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("\n--- Model Evaluation Metrics ---")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")
    
    # Save Model
    model_path = os.path.join(model_dir, 'student_score_model.joblib')
    joblib.dump(model, model_path)
    print(f"Saved model to: {model_path}")
    
    # Save Metrics CSV
    metrics_df = pd.DataFrame([
        {'Metric': 'Mean Absolute Error (MAE)', 'Value': round(mae, 4)},
        {'Metric': 'Root Mean Squared Error (RMSE)', 'Value': round(rmse, 4)},
        {'Metric': 'R-squared (R2 Score)', 'Value': round(r2, 4)}
    ])
    metrics_df.to_csv(os.path.join(output_dir, 'model_metrics.csv'), index=False)
    
    # Save Test Predictions CSV
    test_results = X_test.copy()
    test_results['Actual_Score'] = y_test
    test_results['Predicted_Score'] = np.round(y_pred, 1)
    test_results.to_csv(os.path.join(output_dir, 'test_predictions.csv'), index=False)
    print("Saved evaluation outputs.")

if __name__ == '__main__':
    train()
