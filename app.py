from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the model and scaler from the 'models' folder
model = joblib.load(os.path.join('models', 'best_model.pkl'))
scaler = joblib.load(os.path.join('models', 'scaler.pkl'))

# These must match EXACTLY the order used in training
FEATURES = [
    'Gender', 'Married', 'Dependents', 'Education', 
    'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 
    'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area'
]

@app.route('/')
def home():
    # Serve the HTML page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the frontend
        data = request.get_json()
        
        # Extract values in the exact feature order
        input_values = [float(data[f]) for f in FEATURES]
        
        # Scale the input
        input_scaled = scaler.transform([input_values])
        
        # Get probability of default (class 1)
        prob_default = model.predict_proba(input_scaled)[0][1]
        
        # Determine risk level
        if prob_default < 0.3:
            risk = '🟢 Low Risk'
        elif prob_default < 0.6:
            risk = '🟡 Medium Risk'
        else:
            risk = '🔴 High Risk'
        
        # Send response
        return jsonify({
            'prediction': 'Default' if prob_default >= 0.5 else 'Repay',
            'probability': round(prob_default * 100, 2),
            'risk_level': risk
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)