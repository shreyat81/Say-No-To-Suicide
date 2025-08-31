#!/usr/bin/env python3
"""
Test API for Risk Meter System (without TensorFlow dependency)
This demonstrates the risk meter integration with simulated emotion predictions.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from risk_meter_module import RiskMeter
import random

# Initialize Flask app and risk meter
app = Flask(__name__)
CORS(app)

# Initialize the risk meter
rm = RiskMeter(
    db_path="test_risk_meter.db",
    thresholds={"warn": 20, "alert": 50, "ban": 80}
)

# Simulated emotion labels (same as your model)
emotion_labels = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

def simulate_emotion_prediction():
    """Simulate emotion prediction for testing"""
    # Create random probabilities that sum to 1
    probs = [random.random() for _ in emotion_labels]
    total = sum(probs)
    probs = [p/total for p in probs]
    
    # Create emotion probability dictionary
    emotion_probs = {}
    for i, label in enumerate(emotion_labels):
        emotion_probs[label] = probs[i]
    
    # Get the predicted emotion (highest probability)
    predicted_label = max(emotion_probs, key=emotion_probs.get)
    confidence = emotion_probs[predicted_label]
    
    return predicted_label, confidence, emotion_probs

@app.route('/')
def home():
    return jsonify({
        'message': 'Speech Emotion Recognition Risk Meter Test API',
        'endpoints': {
            '/predict_test': 'POST - Test emotion prediction with risk assessment',
            '/user_status/<user_id>': 'GET - Get user risk status',
            '/admin/users': 'GET - List all users',
            '/reset_user/<user_id>': 'POST - Reset user risk score'
        }
    })

@app.route('/predict_test', methods=['POST'])
def predict_test():
    """Test endpoint that simulates emotion prediction with risk assessment"""
    
    # Get form data
    user_id = request.form.get('user_id', 'test_user')
    user_email = request.form.get('user_email')
    user_text = request.form.get('user_text', '')
    emotion_override = request.form.get('emotion_override')  # For testing specific emotions
    
    try:
        # Simulate emotion prediction
        if emotion_override and emotion_override in emotion_labels:
            # Use override for testing
            predicted_label = emotion_override
            confidence = 0.85
            emotion_probs = {label: 0.1 if label != emotion_override else 0.85 for label in emotion_labels}
            # Normalize
            total = sum(emotion_probs.values())
            emotion_probs = {k: v/total for k, v in emotion_probs.items()}
        else:
            predicted_label, confidence, emotion_probs = simulate_emotion_prediction()
        
        # Update risk meter with prediction
        risk_result = rm.update_with_ser(
            user_id=user_id,
            ser_probs=emotion_probs,
            email=user_email,
            user_text=user_text
        )
        
        # Check if user is banned
        user_row = rm._get_user_row(user_id)
        is_banned = user_row[3] if user_row else False
        
        response = {
            'emotion': predicted_label,
            'confidence': confidence,
            'emotion_probabilities': emotion_probs,
            'risk_info': {
                'user_id': user_id,
                'old_score': risk_result['old_score'],
                'new_score': risk_result['new_score'],
                'increment': risk_result['increment'],
                'keyword_risk': risk_result['keyword_risk'],
                'action': risk_result['action'],
                'is_banned': bool(is_banned)
            },
            'test_mode': True
        }
        
        if is_banned:
            response['warning'] = '🚨 This account has been flagged for concerning behavior patterns.'
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/user_status/<user_id>', methods=['GET'])
def get_user_status(user_id):
    """Get current risk status for a user"""
    try:
        user_row = rm._get_user_row(user_id)
        if not user_row:
            return jsonify({'error': 'User not found'}), 404
        
        return jsonify({
            'user_id': user_row[0],
            'risk_score': user_row[1],
            'last_update': user_row[2],
            'is_banned': bool(user_row[3]),
            'email': user_row[4]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/reset_user/<user_id>', methods=['POST'])
def reset_user_risk(user_id):
    """Reset a user's risk score (admin function)"""
    try:
        rm.reset_user(user_id)
        return jsonify({'message': f'User {user_id} risk score has been reset'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/admin/users', methods=['GET'])
def list_all_users():
    """List all users and their risk scores (admin function)"""
    try:
        import sqlite3
        conn = sqlite3.connect(rm.db_path)
        cur = conn.cursor()
        cur.execute("SELECT user_id, risk_score, last_update, banned, email FROM users ORDER BY risk_score DESC")
        users = cur.fetchall()
        conn.close()
        
        user_list = []
        for user in users:
            user_list.append({
                'user_id': user[0],
                'risk_score': user[1],
                'last_update': user[2],
                'is_banned': bool(user[3]),
                'email': user[4]
            })
        
        return jsonify({'users': user_list})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test_scenarios', methods=['GET'])
def test_scenarios():
    """Get predefined test scenarios"""
    scenarios = [
        {
            'name': 'Happy User',
            'emotion': 'happy',
            'text': 'I am feeling great today!',
            'expected': 'Should reduce risk score'
        },
        {
            'name': 'Sad User',
            'emotion': 'sad',
            'text': 'I am feeling very down',
            'expected': 'Should increase risk score significantly'
        },
        {
            'name': 'Angry User',
            'emotion': 'angry',
            'text': 'I am so frustrated with everything',
            'expected': 'Should increase risk score moderately'
        },
        {
            'name': 'High Risk Keywords',
            'emotion': 'neutral',
            'text': 'I want to kill myself, life is worthless',
            'expected': 'Should trigger high keyword risk (+20 points)'
        },
        {
            'name': 'Educational Context',
            'emotion': 'neutral',
            'text': 'Studying suicide prevention for research paper',
            'expected': 'Should reduce keyword impact due to educational context'
        }
    ]
    
    return jsonify({'scenarios': scenarios})

if __name__ == '__main__':
    print("🧪 Risk Meter Test API Starting...")
    print("📊 Risk thresholds: Warn=20, Alert=50, Ban=80")
    print("🔍 Self-harm keyword detection enabled")
    print("📧 Email alerts configured (demo mode)")
    print("\n🌐 Test endpoints available:")
    print("   POST /predict_test - Test emotion prediction")
    print("   GET /user_status/<user_id> - Get user status")
    print("   GET /admin/users - List all users")
    print("   GET /test_scenarios - Get test scenarios")
    print("\n🚀 Server starting on http://localhost:5000")
    app.run(debug=True, port=5001)
