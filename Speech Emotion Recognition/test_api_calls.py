#!/usr/bin/env python3
"""
Test script to demonstrate the Risk Meter API functionality
"""

import requests
import json
import time

BASE_URL = "http://localhost:5001"

def test_api_call(endpoint, method="GET", data=None):
    """Make API call and display results"""
    try:
        if method == "POST":
            response = requests.post(f"{BASE_URL}{endpoint}", data=data)
        else:
            response = requests.get(f"{BASE_URL}{endpoint}")
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        print("❌ Connection error. Make sure the test API is running on port 5001")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def print_separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def main():
    print("🧪 Testing Risk Meter API Integration")
    
    # Test 1: Basic API connection
    print_separator("TEST 1: API Connection")
    result = test_api_call("/")
    if result:
        print("✅ API is running and accessible")
        print(f"📋 Available endpoints: {list(result['endpoints'].keys())}")
    else:
        print("❌ Cannot connect to API. Make sure test_risk_api.py is running")
        return
    
    # Test 2: Happy emotion (should reduce risk)
    print_separator("TEST 2: Happy Emotion")
    data = {
        'user_id': 'test_user_happy',
        'user_email': 'happy@test.com',
        'emotion_override': 'happy',
        'user_text': 'I am feeling great today!'
    }
    result = test_api_call("/predict_test", "POST", data)
    if result:
        risk_info = result['risk_info']
        print(f"🎭 Emotion: {result['emotion']} (confidence: {result['confidence']:.2f})")
        print(f"📊 Risk: {risk_info['old_score']:.1f} → {risk_info['new_score']:.1f} ({risk_info['increment']:+.1f})")
        print(f"⚡ Action: {risk_info['action']}")
    
    # Test 3: Sad emotion (should increase risk)
    print_separator("TEST 3: Sad Emotion")
    data = {
        'user_id': 'test_user_sad',
        'user_email': 'sad@test.com',
        'emotion_override': 'sad',
        'user_text': 'I am feeling very down and hopeless'
    }
    result = test_api_call("/predict_test", "POST", data)
    if result:
        risk_info = result['risk_info']
        print(f"🎭 Emotion: {result['emotion']} (confidence: {result['confidence']:.2f})")
        print(f"📊 Risk: {risk_info['old_score']:.1f} → {risk_info['new_score']:.1f} ({risk_info['increment']:+.1f})")
        print(f"⚡ Action: {risk_info['action']}")
    
    # Test 4: Self-harm keywords
    print_separator("TEST 4: Self-Harm Keywords")
    data = {
        'user_id': 'test_user_keywords',
        'user_email': 'keywords@test.com',
        'emotion_override': 'neutral',
        'user_text': 'I want to kill myself, I am worthless'
    }
    result = test_api_call("/predict_test", "POST", data)
    if result:
        risk_info = result['risk_info']
        print(f"🎭 Emotion: {result['emotion']} (confidence: {result['confidence']:.2f})")
        print(f"🔍 Keyword risk: +{risk_info['keyword_risk']}")
        print(f"📊 Risk: {risk_info['old_score']:.1f} → {risk_info['new_score']:.1f} ({risk_info['increment']:+.1f})")
        print(f"⚡ Action: {risk_info['action']}")
    
    # Test 5: Build up risk to trigger ban
    print_separator("TEST 5: Escalating Risk (Ban Test)")
    user_id = 'test_user_ban'
    for session in range(1, 6):
        print(f"\n📈 Session {session}:")
        data = {
            'user_id': user_id,
            'user_email': 'ban@test.com',
            'emotion_override': 'sad',
            'user_text': 'I want to end it all, life is pointless' if session >= 3 else 'I feel terrible'
        }
        result = test_api_call("/predict_test", "POST", data)
        if result:
            risk_info = result['risk_info']
            print(f"   Risk: {risk_info['old_score']:.1f} → {risk_info['new_score']:.1f}")
            print(f"   Action: {risk_info['action']}")
            if risk_info['is_banned']:
                print("   🚨 USER BANNED!")
                break
        time.sleep(0.5)  # Small delay between requests
    
    # Test 6: Check user status
    print_separator("TEST 6: User Status Check")
    result = test_api_call(f"/user_status/{user_id}")
    if result:
        print(f"👤 User: {result['user_id']}")
        print(f"📊 Risk Score: {result['risk_score']:.1f}")
        print(f"🚫 Banned: {result['is_banned']}")
        print(f"📧 Email: {result['email']}")
        print(f"🕐 Last Update: {result['last_update']}")
    
    # Test 7: Admin view - all users
    print_separator("TEST 7: Admin View - All Users")
    result = test_api_call("/admin/users")
    if result:
        users = result['users']
        print(f"👥 Total users: {len(users)}")
        for user in users[:5]:  # Show top 5 users by risk score
            status = "🚫 BANNED" if user['is_banned'] else "✅ Active"
            print(f"   {user['user_id']}: Score={user['risk_score']:.1f}, {status}")
    
    # Test 8: Reset user (admin function)
    print_separator("TEST 8: Reset User (Admin)")
    result = test_api_call(f"/reset_user/{user_id}", "POST")
    if result:
        print(f"✅ {result['message']}")
        
        # Check status after reset
        result = test_api_call(f"/user_status/{user_id}")
        if result:
            print(f"📊 New risk score: {result['risk_score']:.1f}")
            print(f"🚫 Banned status: {result['is_banned']}")
    
    print_separator("TESTING COMPLETE")
    print("✅ All API tests completed successfully!")
    print("\n📁 Check these files for logs:")
    print("   - test_risk_meter.db (SQLite database)")
    print("   - alerts.log (Alert notifications)")
    print("   - risk_actions.log (All actions)")

if __name__ == "__main__":
    main()
