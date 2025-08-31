from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import librosa
import numpy as np
import tensorflow as tf
import os
from werkzeug.utils import secure_filename
from risk_meter_module import RiskMeter
from suicide_detection import analyze_text_for_suicide_risk


#from keras.models import load_model
#from keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Bidirectional

#model = load_model("model.h5", custom_objects={
#     "LSTM": LSTM,
#     "Dense": Dense,
#     "Dropout": Dropout,
#     "Conv1D": Conv1D,
#     "MaxPooling1D": MaxPooling1D,
#     "Flatten": Flatten,
#     "Bidirectional": Bidirectional
# })

# Initialize the risk meter with enhanced configuration
rm = RiskMeter(
    db_path="risk_meter.db",
    thresholds={"warn": 20, "alert": 50, "ban": 80},
    smtp_config={
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "sender_email": "alerts@yourcompany.com",
        "sender_password": "your_app_password",
        "admin_email": "admin@yourcompany.com"
    }
)
app = Flask(__name__)
CORS(app)

# Load model and labels
from tensorflow.keras.models import load_model
model = load_model("model.h5")  # Or .keras if you used that format

# Compile if needed (especially for prediction)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

emotion_labels = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

def extract_features(audio_path):
    SAMPLE_RATE = 22050
    DURATION = 4
    FIXED_LENGTH = 173
    N_MFCC = 40

    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION)
    if len(y) < SAMPLE_RATE * DURATION:
        padding = SAMPLE_RATE * DURATION - len(y)
        y = np.pad(y, (0, padding), mode='constant')

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-10)

    if mfcc.shape[1] < FIXED_LENGTH:
        pad_width = FIXED_LENGTH - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :FIXED_LENGTH]

    return mfcc.T[np.newaxis, :, :]

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
        inactivity_days = request.json.get('inactivity_days', 7)
        decay_amount = request.json.get('decay_amount', 5.0)
        rm.decay_scores(inactivity_days=inactivity_days, decay_amount=decay_amount)
        return jsonify({'message': 'Score decay completed'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_text', methods=['POST'])
def analyze_text_only():
    """Endpoint for text-only suicide risk analysis"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data.get('text', '').strip()
        user_id = data.get('user_id', 'anonymous_user')
        user_email = data.get('user_email')
        
        if not text:
            return jsonify({'error': 'Empty text provided'}), 400
        
        # Analyze text for suicide risk
        suicide_analysis = analyze_text_for_suicide_risk(text)
        
        # Update risk meter based on suicide analysis
        if suicide_analysis and 'score' in suicide_analysis:
            # Convert suicide score to emotion-like probabilities for risk meter
            suicide_score = suicide_analysis['score']
            risk_increment = max(0, (suicide_score - 1) * 10)  # Scale 1-10 to 0-90
            
            # Create mock emotion probabilities weighted by suicide risk
            if suicide_score > 6:
                emotion_probs = {'sad': 0.4, 'angry': 0.3, 'fearful': 0.2, 'neutral': 0.1}
            elif suicide_score > 4:
                emotion_probs = {'sad': 0.3, 'neutral': 0.3, 'angry': 0.2, 'fearful': 0.2}
            else:
                emotion_probs = {'neutral': 0.4, 'calm': 0.3, 'happy': 0.2, 'sad': 0.1}
        else:
            emotion_probs = {'neutral': 1.0}
            risk_increment = 0
        
        # Update risk meter
        risk_result = rm.update_with_ser(
            user_id=user_id,
            ser_probs=emotion_probs,
            email=user_email,
            user_text=text
        )
        
        # Check if user is banned
        user_row = rm._get_user_row(user_id)
        is_banned = user_row[3] if user_row else False
        
        response = {
            'suicide_analysis': suicide_analysis,
            'risk_info': {
                'user_id': user_id,
                'old_score': risk_result['old_score'],
                'new_score': risk_result['new_score'],
                'increment': risk_result['increment'],
                'keyword_risk': risk_result['keyword_risk'],
                'action': risk_result['action'],
                'is_banned': bool(is_banned)
            },
            'text_analyzed': text
        }
        
        if is_banned:
            response['warning'] = '🚨 This account has been flagged for concerning behavior patterns.'
        
        # Add additional warnings based on suicide analysis
        if suicide_analysis and suicide_analysis.get('risk_level') == 'critical':
            response['critical_warning'] = '⚠️ CRITICAL: High suicide risk detected. Immediate intervention recommended.'
        elif suicide_analysis and suicide_analysis.get('risk_level') == 'high':
            response['high_warning'] = '⚠️ HIGH RISK: Concerning suicide-related content detected.'
        
        return jsonify(response)
        
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






@app.route('/predict_test', methods=['POST'])
def predict_test():
    """Test endpoint that returns mock emotion data without requiring the ML model"""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    file = request.files['audio']
    user_id = request.form.get('user_id', 'anonymous_user')
    user_email = request.form.get('user_email')
    user_text = request.form.get('user_text', '')
    
    # Mock emotion prediction for testing
    import random
    mock_emotions = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    predicted_label = random.choice(mock_emotions)
    confidence = random.uniform(0.6, 0.95)
    
    # Create mock probability dictionary
    emotion_probs = {}
    remaining_prob = 1.0 - confidence
    for i, label in enumerate(mock_emotions):
        if label == predicted_label:
            emotion_probs[label] = confidence
        else:
            emotion_probs[label] = remaining_prob / (len(mock_emotions) - 1)
    
    try:
        # Analyze text for suicide risk if provided
        suicide_analysis = None
        if user_text and user_text.strip():
            suicide_analysis = analyze_text_for_suicide_risk(user_text)
        
        # Update risk meter with mock prediction
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
            'suicide_analysis': suicide_analysis,
            'test_mode': True
        }
        
        if is_banned:
            response['warning'] = '🚨 This account has been flagged for concerning behavior patterns.'
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    file = request.files['audio']
    user_id = request.form.get('user_id', 'anonymous_user')
    user_email = request.form.get('user_email')
    user_text = request.form.get('user_text', '')  # Optional text input for keyword scanning
    
    filename = secure_filename(file.filename)
    filepath = os.path.join('uploads', filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(filepath)

    try:
        # Extract features and predict emotion
        features = extract_features(filepath)
        prediction = model.predict(features)
        predicted_label = emotion_labels[np.argmax(prediction)]
        confidence = float(np.max(prediction))
        
        # Create probability dictionary for risk meter
        emotion_probs = {}
        for i, label in enumerate(emotion_labels):
            emotion_probs[label] = float(prediction[0][i])
        
        # Analyze text for suicide risk if provided
        suicide_analysis = None
        if user_text and user_text.strip():
            suicide_analysis = analyze_text_for_suicide_risk(user_text)
        
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
            'suicide_analysis': suicide_analysis
        }
        
        if is_banned:
            response['warning'] = '🚨 This account has been flagged for concerning behavior patterns.'
        
        return jsonify(response)
        
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🎤 Speech Emotion Recognition with Risk Monitoring System")
    print("📊 Risk thresholds: Warn=20, Alert=50, Ban=80")
    print("🔍 Self-harm keyword detection enabled")
    print("📧 Email alerts configured (demo mode)")
    print("\n🌐 Available endpoints:")
    print("   / - Home page")
    print("   /demo - Demo page")
    print("   /risk_dashboard - Risk monitoring dashboard")
    print("   /admin_panel - Admin control panel")
    print("   /predict - Emotion prediction with risk assessment")
    print("   /predict_test - Test endpoint with mock emotion data")
    print("   /analyze_text - Text-only suicide risk analysis")
    print("\n🌐 Server starting...")
    app.run(debug=True) 