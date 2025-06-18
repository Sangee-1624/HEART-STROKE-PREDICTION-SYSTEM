from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import pickle
import os

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load and preprocess the data
def load_and_preprocess_data():
    df = pd.read_csv('healthcare-dataset-stroke-data.csv')
    imputer = SimpleImputer(strategy='mean')
    df['bmi'] = imputer.fit_transform(df[['bmi']])
    
    categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    le_dict = {}
    
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le
    
    X = df.drop(['id', 'stroke'], axis=1)
    y = df['stroke']
    
    return X, y, le_dict

# Train the model
def train_model():
    X, y, le_dict = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    with open('rf_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    with open('label_encoders.pkl', 'wb') as f:
        pickle.dump(le_dict, f)
    
    return rf_model, le_dict

# Load or train model
if os.path.exists('rf_model.pkl') and os.path.exists('label_encoders.pkl'):
    with open('rf_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('label_encoders.pkl', 'rb') as f:
        le_dict = pickle.load(f)
else:
    model, le_dict = train_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_data = pd.DataFrame({
            'gender': [data['gender']],
            'age': [float(data['age'])],
            'hypertension': [int(data['hypertension'])],
            'heart_disease': [int(data['heart_disease'])],
            'ever_married': [data['ever_married']],
            'work_type': [data['work_type']],
            'Residence_type': [data['Residence_type']],
            'avg_glucose_level': [float(data['avg_glucose_level'])],
            'bmi': [float(data['bmi'])],
            'smoking_status': [data['smoking_status']]
        })
        
        categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        for col in categorical_columns:
            input_data[col] = le_dict[col].transform(input_data[col])
        
        prediction = model.predict_proba(input_data)[0]
        stroke_risk = prediction[1] * 100
        
        return jsonify({
            'stroke_risk': round(stroke_risk, 2),
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)