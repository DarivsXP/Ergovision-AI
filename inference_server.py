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
        
        # Validation: Ensure we actually got data
        if not data or 'landmarks' not in data:
            return jsonify({'error': 'No landmarks provided'}), 400

        landmarks = data['landmarks']
        ideal_back = float(data.get('ideal_back', 0))
        ideal_neck = float(data.get('ideal_neck', 0))
        
        # Calculate Angles
        # Note: MediaPipe landmarks are objects, so accessing ['x'] is correct
        # if you sent them as JSON.
        ear = [landmarks[8]['x'], landmarks[8]['y']]
        shoulder = [landmarks[12]['x'], landmarks[12]['y']]
        hip = [landmarks[24]['x'], landmarks[24]['y']]

        neck_a = round(calculate_angle(shoulder, ear), 2)
        back_a = round(calculate_angle(hip, shoulder), 2)

        # 1. AI Prediction
        # The model expects a 2D array [[f1, f2, f3]]
        features = scaler.transform([[neck_a, back_a, 0.0]]) 
        prediction = int(model.predict(features)[0])
        
        # 2. Score Calculation
        dev = abs(back_a - ideal_back) + abs(neck_a - ideal_neck)
        calculated_score = 100 - (dev * 2.0) 
        
        # 3. Blending Logic
        if prediction == 1:
            # If AI sees "Slouch", cap the score at 85 (Warning)
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
        print(f"Error processing request: {e}")
        return jsonify({'error': str(e)}), 400  

# 5. START SERVER
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)