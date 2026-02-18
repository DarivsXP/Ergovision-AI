import os
import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# 1. INITIALIZE APP
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# 2. LOAD MODELS
try:
    model = joblib.load('ergovision_final_model.pkl')
    scaler = joblib.load('posture_scaler.pkl')
    print("Models loaded successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load models. {e}")

# 3. HELPER FUNCTIONS
def calculate_angle(p1, p2):
    """Calculates the vertical angle between two points."""
    return abs(np.degrees(np.arctan2(p2[0] - p1[0], p2[1] - p1[1])))

# 4. ROUTES

@app.route('/', methods=['GET'])
def keep_alive():
    return "Ergovision AI is Awake!", 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data or 'landmarks' not in data:
            return jsonify({'error': 'No landmarks provided'}), 400

        landmarks = data['landmarks']
        ideal_back = float(data.get('ideal_back', 0))
        ideal_neck = float(data.get('ideal_neck', 0))
        is_calibrating = data.get('is_calibrating', False) # Check if user is in calibration mode
        
        # --- 1. VISIBILITY VALIDATION ---
        required_ids = [8, 12, 24] # Ear, Shoulder, Hip
        for l_id in required_ids:
            visibility = landmarks[l_id].get('visibility', 0)
            if visibility < 0.5:
                return jsonify({
                    'label': 0, 'score': 0,
                    'status': 'user_not_detected',
                    'message': 'Please align yourself with the camera'
                })

        # --- 2. EXTRACT COORDINATES ---
        ear = [landmarks[8]['x'], landmarks[8]['y']]
        shoulder = [landmarks[12]['x'], landmarks[12]['y']]
        hip = [landmarks[24]['x'], landmarks[24]['y']]

        # --- 3. STANDING DETECTION ---
        if hip[1] < 0.4: 
             return jsonify({
                'label': 0, 'score': 100,
                'status': 'standing',
                'message': 'Standing detected - Posture tracking paused'
            })

        # --- 4. CALCULATE ANGLES ---
        neck_a = round(calculate_angle(shoulder, ear), 2)
        back_a = round(calculate_angle(hip, shoulder), 2)

        # --- 5. AI PREDICTION ---
        features = scaler.transform([[neck_a, back_a, 0.0]]) 
        prediction = int(model.predict(features)[0]) # 1 = Slouch, 0 = Good
        
        # --- 6. CALIBRATION GUARD ---
        # If user is trying to calibrate but the AI sees a slouch, reject it.
        if is_calibrating and prediction == 1:
            return jsonify({
                'status': 'calibration_error',
                'message': 'Slouch detected! Sit straight to calibrate.'
            })

        # --- 7. DYNAMIC SCORING (The "Alive" Formula) ---
        dev = abs(back_a - ideal_back) + abs(neck_a - ideal_neck)
        calculated_score = 100 - (dev * 0.8) 
        
        # --- 8. SLOUCH OVERRIDE ---
        if prediction == 1:
            final_score = min(calculated_score, 88)
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
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)