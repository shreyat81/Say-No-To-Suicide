
"""
risk_meter_module.py

A simple RiskMeter implementation to integrate with an existing SER model.
- Stores per-user risk scores in a lightweight SQLite DB
- Updates score based on model (emotion) probabilities
- Applies decay (resetting toward zero after inactivity)
- Triggers configurable actions (warning, alert, ban) -- currently logged, not emailing by default
"""

import sqlite3, time, datetime, json, os, smtplib, re
from typing import Dict, Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

DEFAULT_WEIGHTS = {
    "sad": 5.0,
    "disgust": 3.0,
    "angry": 2.0,
    "fearful": 4.0,
    "fear": 4.0,
    "happy": -2.0,
    "neutral": 0.0,
    "calm": 0.0,
    "surprised": 0.0
}

DEFAULT_THRESHOLDS = {"warn": 20.0, "alert": 50.0, "ban": 80.0}

# Self-harm keywords for enhanced detection
SELF_HARM_KEYWORDS = {
    "suicide", "kill myself", "end my life", "want to die", "self harm", "cut myself",
    "poison", "overdose", "jump off", "hang myself", "worthless", "better off dead",
    "no point living", "can't go on", "end it all", "hurt myself"
}

class RiskMeter:
    def __init__(self, db_path: str = "risk_meter.db",
                 weights: Dict[str, float] = None,
                 thresholds: Dict[str, float] = None,
                 educational_terms: Optional[set] = None,
                 smtp_config: Optional[Dict] = None):
        self.db_path = db_path
        self.weights = weights or DEFAULT_WEIGHTS
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self.educational_terms = educational_terms or {"study", "research", "prevention", "prevention", "awareness", "paper", "report"}
        self.smtp_config = smtp_config or {
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "sender_email": "your_alert_email@gmail.com",
            "sender_password": "your_app_password",  # Use app password for Gmail
            "admin_email": "admin@yourcompany.com"
        }
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                risk_score REAL DEFAULT 0,
                last_update TEXT,
                banned INTEGER DEFAULT 0,
                email TEXT
            )
            """
        )
        conn.commit()
        conn.close()

    def _get_user_row(self, user_id: str):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT user_id, risk_score, last_update, banned, email FROM users WHERE user_id = ?", (user_id,))
        row = cur.fetchone()
        conn.close()
        return row

    def ensure_user(self, user_id: str, email: Optional[str] = None):
        if self._get_user_row(user_id) is None:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute("INSERT INTO users (user_id, risk_score, last_update, banned, email) VALUES (?, ?, ?, ?, ?)", 
                        (user_id, 0.0, datetime.datetime.utcnow().isoformat(), 0, email))
            conn.commit()
            conn.close()

    def compute_increment(self, ser_probs: Dict[str, float], educational: bool = False) -> float:
        inc = 0.0
        for emo, prob in ser_probs.items():
            w = self.weights.get(emo.lower(), 0.0)
            inc += w * float(prob)
        # if educational terms present in text, caller should set educational=True to reduce impact
        if educational:
            inc = max(0.0, inc - 3.0)
        return inc

    def scan_for_self_harm_keywords(self, text: str) -> int:
        """Scan text for self-harm keywords and return additional risk points"""
        if not text:
            return 0
        
        text_lower = text.lower()
        keyword_count = 0
        
        for keyword in SELF_HARM_KEYWORDS:
            if keyword in text_lower:
                keyword_count += 1
        
        # Each keyword adds significant risk
        return keyword_count * 10
    
    def update_with_ser(self, user_id: str, ser_probs: Dict[str, float], educational: bool = False, 
                       email: Optional[str] = None, user_text: Optional[str] = None):
        """
        ser_probs: dict of emotion -> probability (values in [0,1])
        user_text: optional text input to scan for self-harm keywords
        """
        self.ensure_user(user_id, email)
        inc = self.compute_increment(ser_probs, educational=educational)
        
        # Add additional risk for self-harm keywords
        keyword_risk = 0
        if user_text:
            keyword_risk = self.scan_for_self_harm_keywords(user_text)
            inc += keyword_risk
        
        now = datetime.datetime.utcnow().isoformat()
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT risk_score FROM users WHERE user_id = ?", (user_id,))
        row = cur.fetchone()
        current = row[0] if row else 0.0
        new_score = max(0.0, current + inc)
        # persist
        cur.execute("UPDATE users SET risk_score = ?, last_update = ?, email = ? WHERE user_id = ?", (new_score, now, email, user_id))
        conn.commit()
        conn.close()
        action = self._check_thresholds_and_act(user_id, new_score, email)
        return {
            "old_score": current, 
            "increment": inc, 
            "keyword_risk": keyword_risk,
            "new_score": new_score, 
            "action": action
        }

    def _check_thresholds_and_act(self, user_id: str, score: float, email: Optional[str] = None):
        # Returns one of: "none", "warn", "alert", "ban"
        if score >= self.thresholds.get("ban", 80.0):
            self._mark_banned(user_id)
            self._send_alert_email(user_id, "ACCOUNT BANNED", score, email)
            self._log_action(user_id, "ban", score)
            print(f"🚨 ACCOUNT BANNED: User {user_id} has been banned due to high risk score: {score}")
            return "ban"
        elif score >= self.thresholds.get("alert", 50.0):
            self._send_alert_email(user_id, "HIGH RISK ALERT", score, email)
            self._log_action(user_id, "alert", score)
            return "alert"
        elif score >= self.thresholds.get("warn", 20.0):
            self._log_action(user_id, "warn", score)
            return "warn"
        else:
            return "none"

    def _mark_banned(self, user_id: str):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("UPDATE users SET banned = 1 WHERE user_id = ?", (user_id,))
        conn.commit()
        conn.close()

    def _send_alert_email(self, user_id: str, alert_type: str, score: float, user_email: Optional[str] = None):
        """Send alert email to admin and optionally to user"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.smtp_config['sender_email']
            msg['To'] = self.smtp_config['admin_email']
            msg['Subject'] = f"🚨 Risk Alert: {alert_type} - User {user_id}"
            
            body = f"""
            RISK ALERT NOTIFICATION
            
            Alert Type: {alert_type}
            User ID: {user_id}
            Risk Score: {score}
            Timestamp: {datetime.datetime.utcnow().isoformat()}
            User Email: {user_email or 'Not provided'}
            
            Please take immediate action if this is a ban alert.
            
            This is an automated message from the Speech Emotion Recognition Risk Monitoring System.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Note: In production, you would actually send the email
            # For demo purposes, we'll just log it
            print(f"📧 EMAIL ALERT: {alert_type} for user {user_id} (Score: {score})")
            print(f"   Would send to: {self.smtp_config['admin_email']}")
            
            # Uncomment below for actual email sending:
            # server = smtplib.SMTP(self.smtp_config['smtp_server'], self.smtp_config['smtp_port'])
            # server.starttls()
            # server.login(self.smtp_config['sender_email'], self.smtp_config['sender_password'])
            # server.send_message(msg)
            # server.quit()
            
        except Exception as e:
            print(f"Failed to send alert email: {e}")
    
    def _log_action(self, user_id: str, action: str, score: float):
        # Enhanced logging with more details
        timestamp = datetime.datetime.utcnow().isoformat()
        log_line = f"{timestamp} | ACTION: {action.upper()} | user: {user_id} | score: {score}\n"
        
        # Log to alerts.log as requested
        with open("alerts.log", "a") as f:
            f.write(log_line)
        
        # Also keep the old log file for compatibility
        with open("risk_actions.log", "a") as f:
            f.write(log_line)
            
        print(log_line.strip())

    def get_user_score(self, user_id: str) -> float:
        row = self._get_user_row(user_id)
        if not row:
            return 0.0
        return float(row[1])

    def decay_scores(self, inactivity_days: int = 7, decay_amount: float = 5.0):
        """
        Reduce risk_score for users who haven't updated within inactivity_days.
        If after decay score <= 0, set to zero.
        This function is intended to be run periodically (e.g., daily cron job).
        """
        cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=inactivity_days)
        cutoff_iso = cutoff.isoformat()
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT user_id, risk_score, last_update FROM users")
        rows = cur.fetchall()
        for user_id, score, last_update in rows:
            if last_update is None:
                continue
            try:
                last_dt = datetime.datetime.fromisoformat(last_update)
            except Exception:
                continue
            if last_dt < cutoff:
                new_score = max(0.0, score - decay_amount)
                cur.execute("UPDATE users SET risk_score = ?, last_update = ? WHERE user_id = ?", (new_score, datetime.datetime.utcnow().isoformat(), user_id))
                if new_score == 0.0:
                    # log for visibility
                    with open("risk_actions.log", "a") as f:
                        f.write(f"{datetime.datetime.utcnow().isoformat()} | DECAY_RESET | user: {user_id}\\n")
        conn.commit()
        conn.close()

    def reset_user(self, user_id: str):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("UPDATE users SET risk_score = 0.0, banned = 0, last_update = ? WHERE user_id = ?", (datetime.datetime.utcnow().isoformat(), user_id))
        conn.commit()
        conn.close()

if __name__ == "__main__":
    # quick demonstration
    rm = RiskMeter(db_path="demo_risk.db")
    user = "demo_user_1"
    # simulate three utterances with sad/disgust predictions
    s1 = {"sad": 0.6, "disgust": 0.0, "angry": 0.0, "happy": 0.0, "neutral": 0.4}
    print("Update 1:", rm.update_with_ser(user, s1))
    s2 = {"sad": 0.7, "disgust": 0.2, "angry": 0.0, "happy": 0.0, "neutral": 0.1}
    print("Update 2:", rm.update_with_ser(user, s2))
    s3 = {"sad": 0.8, "disgust": 0.5, "angry": 0.2, "happy": 0.0, "neutral": 0.0}
    print("Update 3:", rm.update_with_ser(user, s3))
