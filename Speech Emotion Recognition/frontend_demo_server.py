#!/usr/bin/env python3
"""
Frontend Demo Server - Complete Risk Meter System
This server demonstrates all frontend features without TensorFlow dependencies
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from risk_meter_module import RiskMeter
import random
import time

# Initialize Flask app and risk meter
app = Flask(__name__)
CORS(app)

# Initialize the risk meter
rm = RiskMeter(
    db_path="frontend_demo.db",
    thresholds={"warn": 20, "alert": 50, "ban": 80}
)

# Simulated emotion labels
emotion_labels = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

def simulate_emotion_prediction(emotion_override=None):
    """Simulate emotion prediction for demo"""
    if emotion_override and emotion_override in emotion_labels:
        # Use specific emotion for testing
        predicted_label = emotion_override
        confidence = 0.85
        emotion_probs = {label: 0.05 if label != emotion_override else 0.85 for label in emotion_labels}
    else:
        # Create random probabilities that sum to 1
        probs = [random.random() for _ in emotion_labels]
        total = sum(probs)
        probs = [p/total for p in probs]
        
        emotion_probs = {}
        for i, label in enumerate(emotion_labels):
            emotion_probs[label] = probs[i]
        
        predicted_label = max(emotion_probs, key=emotion_probs.get)
        confidence = emotion_probs[predicted_label]
    
    # Normalize probabilities
    total = sum(emotion_probs.values())
    emotion_probs = {k: v/total for k, v in emotion_probs.items()}
    
    return predicted_label, confidence, emotion_probs

# Routes for all pages
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/demo')
def demo():
    return render_template('demo.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/technical')
def technical():
    return render_template('technical.html')

@app.route('/risk_dashboard')
def risk_dashboard():
    return render_template('risk_dashboard.html')

@app.route('/admin_panel')
def admin_panel():
    return render_template('admin_panel.html')

# API Endpoints
@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint with risk assessment"""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    file = request.files['audio']
    user_id = request.form.get('user_id', 'demo_user')
    user_email = request.form.get('user_email')
    user_text = request.form.get('user_text', '')
    emotion_override = request.form.get('emotion_override')  # For testing
    
    filename = secure_filename(file.filename)
    filepath = os.path.join('uploads', filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(filepath)

    try:
        # Simulate emotion prediction
        predicted_label, confidence, emotion_probs = simulate_emotion_prediction(emotion_override)
        
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
        
        os.remove(filepath)
        
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
            'demo_mode': True
        }
        
        if is_banned:
            response['warning'] = '🚨 This account has been flagged for concerning behavior patterns.'
        
        return jsonify(response)
        
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500

@app.route('/predict_test', methods=['POST'])
def predict_test():
    """Test endpoint for frontend demo"""
    user_id = request.form.get('user_id', 'demo_user')
    user_email = request.form.get('user_email')
    user_text = request.form.get('user_text', '')
    emotion_override = request.form.get('emotion_override')

    try:
        # Simulate emotion prediction
        predicted_label, confidence, emotion_probs = simulate_emotion_prediction(emotion_override)
        
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

@app.route('/decay_scores', methods=['POST'])
def trigger_decay():
    """Manually trigger score decay (admin function)"""
    try:
        data = request.get_json() or {}
        inactivity_days = data.get('inactivity_days', 7)
        decay_amount = data.get('decay_amount', 5.0)
        rm.decay_scores(inactivity_days=inactivity_days, decay_amount=decay_amount)
        return jsonify({'message': 'Score decay completed'})
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
    """Get predefined test scenarios for frontend demo"""
    scenarios = [
        {
            'name': 'Happy User',
            'user_id': 'happy_user',
            'emotion': 'happy',
            'text': 'I am feeling great today!',
            'expected': 'Should reduce risk score'
        },
        {
            'name': 'Sad User',
            'user_id': 'sad_user',
            'emotion': 'sad',
            'text': 'I am feeling very down',
            'expected': 'Should increase risk score significantly'
        },
        {
            'name': 'Angry User',
            'user_id': 'angry_user',
            'emotion': 'angry',
            'text': 'I am so frustrated with everything',
            'expected': 'Should increase risk score moderately'
        },
        {
            'name': 'High Risk Keywords',
            'user_id': 'keyword_user',
            'emotion': 'neutral',
            'text': 'I want to kill myself, life is worthless',
            'expected': 'Should trigger high keyword risk (+20 points)'
        },
        {
            'name': 'Educational Context',
            'user_id': 'research_user',
            'emotion': 'neutral',
            'text': 'Studying suicide prevention for research paper',
            'expected': 'Should reduce keyword impact due to educational context'
        }
    ]
    
    return jsonify({'scenarios': scenarios})

if __name__ == '__main__':
    print("🎨 Frontend Demo Server Starting...")
    print("📊 Risk thresholds: Warn=20, Alert=50, Ban=80")
    print("🔍 Self-harm keyword detection enabled")
    print("📧 Email alerts configured (demo mode)")
    print("\n🌐 Available pages:")
    print("   http://localhost:5002/ - Home page")
    print("   http://localhost:5002/demo - Demo page")
    print("   http://localhost:5002/risk_dashboard - Risk monitoring dashboard")
    print("   http://localhost:5002/admin_panel - Admin control panel")
    print("\n🚀 Server starting on http://localhost:5002")
    app.run(debug=True, port=5002)
