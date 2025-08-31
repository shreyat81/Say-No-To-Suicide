#!/usr/bin/env python3
"""
Demo script for the Enhanced Speech Emotion Recognition Risk Meter System

This script demonstrates:
1. Risk scoring based on emotion predictions
2. Self-harm keyword detection
3. Threshold-based alerts and banning
4. Score decay mechanism
5. User management functions
"""

from risk_meter_module import RiskMeter
import time

def print_separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def demo_basic_risk_scoring():
    print_separator("DEMO 1: Basic Risk Scoring")
    
    # Initialize risk meter
    rm = RiskMeter(db_path="demo_risk.db")
    
    # Test user
    user_id = "test_user_1"
    
    print(f"Testing user: {user_id}")
    print(f"Initial risk score: {rm.get_user_score(user_id)}")
    
    # Simulate different emotion predictions
    scenarios = [
        {
            "name": "Happy emotion",
            "probs": {"happy": 0.8, "neutral": 0.2, "sad": 0.0, "angry": 0.0, "disgust": 0.0}
        },
        {
            "name": "Sad emotion (moderate)",
            "probs": {"sad": 0.6, "neutral": 0.4, "happy": 0.0, "angry": 0.0, "disgust": 0.0}
        },
        {
            "name": "Very sad emotion",
            "probs": {"sad": 0.9, "neutral": 0.1, "happy": 0.0, "angry": 0.0, "disgust": 0.0}
        },
        {
            "name": "Angry emotion",
            "probs": {"angry": 0.7, "neutral": 0.3, "happy": 0.0, "sad": 0.0, "disgust": 0.0}
        },
        {
            "name": "Mixed negative emotions",
            "probs": {"sad": 0.4, "angry": 0.3, "disgust": 0.2, "neutral": 0.1, "happy": 0.0}
        }
    ]
    
    for scenario in scenarios:
        print(f"\n📊 Scenario: {scenario['name']}")
        result = rm.update_with_ser(user_id, scenario['probs'])
        print(f"   Score change: {result['old_score']:.1f} → {result['new_score']:.1f} (+{result['increment']:.1f})")
        print(f"   Action triggered: {result['action']}")

def demo_keyword_detection():
    print_separator("DEMO 2: Self-Harm Keyword Detection")
    
    rm = RiskMeter(db_path="demo_risk.db")
    user_id = "test_user_2"
    
    # Test various text inputs with keywords
    test_texts = [
        "I'm feeling a bit down today",
        "I want to kill myself, life is pointless",
        "Studying suicide prevention for my research paper",
        "I can't go on anymore, I'm worthless",
        "This is just a normal conversation about emotions"
    ]
    
    # Neutral emotion for all tests to focus on keyword impact
    neutral_emotion = {"neutral": 1.0, "sad": 0.0, "angry": 0.0, "disgust": 0.0, "happy": 0.0}
    
    print(f"Testing user: {user_id}")
    print(f"Initial risk score: {rm.get_user_score(user_id)}")
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n🔍 Test {i}: '{text}'")
        
        # Check if it's educational content
        educational = "research" in text.lower() or "study" in text.lower()
        
        result = rm.update_with_ser(
            user_id, 
            neutral_emotion, 
            educational=educational,
            user_text=text
        )
        
        print(f"   Keyword risk: +{result['keyword_risk']}")
        print(f"   Total increment: +{result['increment']:.1f}")
        print(f"   New score: {result['new_score']:.1f}")
        print(f"   Action: {result['action']}")

def demo_threshold_system():
    print_separator("DEMO 3: Threshold System & Account Banning")
    
    rm = RiskMeter(db_path="demo_risk.db")
    user_id = "high_risk_user"
    
    print(f"Testing user: {user_id}")
    print("Simulating escalating risk behavior...")
    
    # Simulate escalating harmful emotions
    high_risk_emotions = {"sad": 0.8, "angry": 0.6, "disgust": 0.4, "neutral": 0.0, "happy": 0.0}
    
    for session in range(1, 8):
        print(f"\n📈 Session {session}:")
        result = rm.update_with_ser(user_id, high_risk_emotions)
        
        print(f"   Score: {result['old_score']:.1f} → {result['new_score']:.1f}")
        print(f"   Action: {result['action']}")
        
        if result['action'] == 'ban':
            print("   🚨 USER BANNED! No further sessions allowed.")
            break
        
        # Add some self-harm text in later sessions
        if session >= 4:
            keyword_result = rm.update_with_ser(
                user_id, 
                {"neutral": 1.0, "sad": 0.0, "angry": 0.0, "disgust": 0.0, "happy": 0.0},
                user_text="I want to end it all"
            )
            print(f"   + Keyword detection: +{keyword_result['keyword_risk']} risk")

def demo_decay_mechanism():
    print_separator("DEMO 4: Score Decay Mechanism")
    
    rm = RiskMeter(db_path="demo_risk.db")
    user_id = "decay_test_user"
    
    # Build up some risk
    high_risk = {"sad": 0.9, "angry": 0.5, "neutral": 0.0, "happy": 0.0, "disgust": 0.0}
    
    print(f"Building up risk for user: {user_id}")
    for i in range(3):
        result = rm.update_with_ser(user_id, high_risk)
        print(f"Session {i+1}: Score = {result['new_score']:.1f}")
    
    print(f"\nCurrent score before decay: {rm.get_user_score(user_id):.1f}")
    
    # Simulate decay (normally this would be run as a cron job)
    print("\n🕐 Simulating 7 days of inactivity...")
    print("Running decay mechanism...")
    
    rm.decay_scores(inactivity_days=0, decay_amount=10.0)  # Force decay by setting days to 0
    
    print(f"Score after decay: {rm.get_user_score(user_id):.1f}")

def demo_admin_functions():
    print_separator("DEMO 5: Admin Functions")
    
    rm = RiskMeter(db_path="demo_risk.db")
    
    print("📋 All users in system:")
    
    # Get all users (simulating the admin endpoint)
    import sqlite3
    conn = sqlite3.connect(rm.db_path)
    cur = conn.cursor()
    cur.execute("SELECT user_id, risk_score, last_update, banned, email FROM users ORDER BY risk_score DESC")
    users = cur.fetchall()
    conn.close()
    
    for user in users:
        status = "🚫 BANNED" if user[3] else "✅ Active"
        print(f"   {user[0]}: Score={user[1]:.1f}, {status}")
    
    # Reset a user
    if users:
        reset_user = users[0][0]  # Reset the highest risk user
        print(f"\n🔄 Resetting user: {reset_user}")
        rm.reset_user(reset_user)
        print(f"   New score: {rm.get_user_score(reset_user):.1f}")

def main():
    print("🎤 Speech Emotion Recognition Risk Meter System Demo")
    print("📊 Risk Scoring: Sad=+5, Angry=+2, Disgust=+3, Happy=-2")
    print("🔍 Self-harm keywords: +10 each")
    print("⚠️  Thresholds: Warn=20, Alert=50, Ban=80")
    
    try:
        demo_basic_risk_scoring()
        demo_keyword_detection()
        demo_threshold_system()
        demo_decay_mechanism()
        demo_admin_functions()
        
        print_separator("DEMO COMPLETE")
        print("✅ All risk meter features demonstrated successfully!")
        print("\n📁 Check these files:")
        print("   - demo_risk.db (SQLite database)")
        print("   - alerts.log (Alert log file)")
        print("   - risk_actions.log (Action log file)")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
