from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

@app.route('/', methods=['GET'])
def keep_alive():
    return "Ergovision AI is Awake!", 200
# --------------------------

@app.route('/predict', methods=['POST'])
def predict():
    
app = Flask(__name__)
# Optimized CORS for fast local requests
CORS(app, resources={r"/*": {"origins": "*"}})

# Load models
model = joblib.load('ergovision_final_model.pkl')
scaler = joblib.load('posture_scaler.pkl')

def calculate_angle(p1, p2):
    # Vectorized math for high-frequency hits
    return abs(np.degrees(np.arctan2(p2[0] - p1[0], p2[1] - p1[1])))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        landmarks = data['landmarks']
        ideal_back = float(data.get('ideal_back', 0))
        ideal_neck = float(data.get('ideal_neck', 0))
        
        # Calculate Angles
        ear = [landmarks[8]['x'], landmarks[8]['y']]
        shoulder = [landmarks[12]['x'], landmarks[12]['y']]
        hip = [landmarks[24]['x'], landmarks[24]['y']]

        neck_a = round(calculate_angle(shoulder, ear), 2)
        back_a = round(calculate_angle(hip, shoulder), 2)

        # 1. AI Prediction
        features = scaler.transform([[neck_a, back_a, 0.0]]) 
        prediction = int(model.predict(features)[0])
        
        # 2. Score Calculation (THE FIX)
        # Old Multiplier: 3 or 4 (Too Strict)
        # New Multiplier: 2.0 (Forgiving)
        dev = abs(back_a - ideal_back) + abs(neck_a - ideal_neck)
        calculated_score = 100 - (dev * 2.0) 
        
        # 3. Blending Logic (THE FIX)
        # Old Cap: 75% (Always looks like a failure)
        # New Cap: 85% (Looks like a "Warning")
        if prediction == 1:
            final_score = min(calculated_score, 85)
        else:
            final_score = calculated_score

        # Ensure score stays between 0 and 100
        final_score = max(0, min(100, int(final_score)))

        print(f"Pred: {prediction} | Score: {final_score}%")

        return jsonify({
            'label': prediction,
            'score': final_score,
            'angles': {'neck': neck_a, 'back': back_a}
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 400  

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)