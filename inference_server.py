import os
import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# 1. INITIALIZE APP FIRST (The Foundation)
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# 2. LOAD MODELS (Global Scope)
# Make sure these files are in the same folder or update the path
try:
    model = joblib.load('ergovision_final_model.pkl')
    scaler = joblib.load('posture_scaler.pkl')
    print("Models loaded successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load models. {e}")

# 3. HELPER FUNCTIONS
def calculate_angle(p1, p2):
    # Vectorized math for high-frequency hits
    # Ensure points are numpy arrays or lists of floats
    return abs(np.degrees(np.arctan2(p2[0] - p1[0], p2[1] - p1[1])))

# 4. ROUTES

# Health Check (Keep Alive)
@app.route('/', methods=['GET'])
def keep_alive():
    return "Ergovision AI is Awake!", 200

# Prediction Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data or 'landmarks' not in data:
            return jsonify({'error': 'No landmarks provided'}), 400

        landmarks = data['landmarks']
        ideal_back = float(data.get('ideal_back', 0))
        ideal_neck = float(data.get('ideal_neck', 0))
        
        # --- VALIDATION: Is the person actually there? ---
        # MediaPipe provides visibility (0.0 to 1.0)
        # We check Ear (8), Shoulder (12), and Hip (24)
        required_ids = [8, 12, 24]
        for l_id in required_ids:
            # Check if landmark exists and if it's visible enough (> 50%)
            visibility = landmarks[l_id].get('visibility', 0)
            if visibility < 0.5:
                return jsonify({
                    'label': 0,
                    'score': 0,
                    'status': 'user_not_detected',
                    'message': 'Please align yourself with the camera'
                })

        # --- EXTRACT COORDINATES ---
        ear = [landmarks[8]['x'], landmarks[8]['y']]
        shoulder = [landmarks[12]['x'], landmarks[12]['y']]
        hip = [landmarks[24]['x'], landmarks[24]['y']]

        # --- STANDING DETECTION ---
        # If the hip is significantly higher in the frame (lower Y value), 
        # they are likely standing or too close.
        if hip[1] < 0.4: # Adjust 0.4 based on your camera testing
             return jsonify({
                'label': 0,
                'score': 100,
                'status': 'standing',
                'message': 'Standing detected - Posture tracking paused'
            })

        # Calculate Angles
        neck_a = round(calculate_angle(shoulder, ear), 2)
        back_a = round(calculate_angle(hip, shoulder), 2)

        # 1. AI Prediction
        features = scaler.transform([[neck_a, back_a, 0.0]]) 
        prediction = int(model.predict(features)[0])
        
        # 2. Score Calculation (FORGIVING LOGIC)
        dev = abs(back_a - ideal_back) + abs(neck_a - ideal_neck)

        # Buffer increased to 10, Multiplier dropped to 1.1
        if dev < 10:
            calculated_score = 100
        else:
            calculated_score = 100 - ((dev - 10) * 1.1) 
        
        # 3. Blending Logic
        if prediction == 1:
            final_score = min(calculated_score, 75)
        else:
            final_score = calculated_score

        final_score = max(0, min(100, int(final_score)))

        return jsonify({
            'label': prediction,
            'score': final_score,
            'status': 'active',
            'angles': {'neck': neck_a, 'back': back_a}
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400