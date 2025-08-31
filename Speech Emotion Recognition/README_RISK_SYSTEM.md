# Speech Emotion Recognition with Risk Meter System

## 🚨 Enhanced Risk Monitoring Features

This project now includes a comprehensive risk assessment system that monitors user emotional patterns and detects potentially harmful behavior.

### 🎯 Key Features

#### 1. **Risk Scoring System**
- **Harmful emotions** increase risk score:
  - Sad: +5 points
  - Angry: +2 points  
  - Disgust: +3 points
  - Fearful: +4 points
- **Positive emotions** reduce risk score:
  - Happy: -2 points
- **Neutral emotions**: No change

#### 2. **Self-Harm Keyword Detection**
- Scans text input for concerning keywords
- Each keyword adds +10 risk points
- Keywords include: "suicide", "kill myself", "end my life", "worthless", etc.
- Educational context detection reduces impact

#### 3. **Threshold-Based Actions**
- **Warning (20+ points)**: Log warning
- **Alert (50+ points)**: Send email alert to admin
- **Ban (80+ points)**: Account banned + email alert

#### 4. **Score Decay Mechanism**
- Automatically reduces scores for inactive users
- Default: -5 points after 7 days of inactivity
- Prevents permanent penalties for recovered users

#### 5. **Persistent Storage**
- SQLite database stores user data
- Tracks: user_id, risk_score, last_update, banned status, email

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Demo
```bash
python demo_risk_system.py
```

### 3. Start Flask App
```bash
python app.py
```

## 📡 API Endpoints

### Core Prediction
- `POST /predict` - Analyze audio with risk assessment
  - Form data: `audio` (file), `user_id`, `user_email`, `user_text`
  - Returns: emotion + risk info

### User Management
- `GET /user_status/<user_id>` - Get user risk status
- `POST /reset_user/<user_id>` - Reset user risk score (admin)
- `GET /admin/users` - List all users with risk scores

### System Management
- `POST /decay_scores` - Trigger score decay manually

## 🔧 Configuration

### Risk Meter Setup
```python
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
```

### Email Alerts
To enable real email sending, uncomment the SMTP code in `_send_alert_email()` method and configure your email credentials.

## 📊 Usage Example

```python
from risk_meter_module import RiskMeter

# Initialize
rm = RiskMeter()

# Update with emotion prediction
emotion_probs = {"sad": 0.8, "angry": 0.2, "happy": 0.0, "neutral": 0.0}
result = rm.update_with_ser(
    user_id="user123",
    ser_probs=emotion_probs,
    user_text="I feel worthless",
    email="user@example.com"
)

print(f"Risk score: {result['new_score']}")
print(f"Action: {result['action']}")
```

## 📁 Generated Files

- `risk_meter.db` - SQLite database with user data
- `alerts.log` - Alert notifications log
- `risk_actions.log` - All risk actions log

## ⚠️ Important Notes

1. **Privacy**: This system processes sensitive emotional data
2. **Ethics**: Use responsibly with proper user consent
3. **Testing**: Run `demo_risk_system.py` to test all features
4. **Production**: Configure real SMTP for email alerts
5. **Monitoring**: Regularly review alert logs

## 🛡️ Safety Features

- Educational content detection reduces false positives
- Gradual score decay prevents permanent penalties
- Multiple threshold levels for graduated response
- Comprehensive logging for audit trails
- Admin functions for user management

## 📞 Support

For issues or questions about the risk monitoring system, check the logs and ensure all dependencies are installed correctly.
