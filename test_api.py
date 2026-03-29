#!/usr/bin/env python3
"""
Test script for API endpoints
"""
import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_health():
    """Test API health"""
    print("=" * 60)
    print("TEST 1: API Health Check")
    print("=" * 60)
    try:
        response = requests.get(f"{BASE_URL}/docs")
        print(f"✅ API Documentation accessible: Status {response.status_code}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_fake_job():
    """Test prediction with fake job"""
    print("\n" + "=" * 60)
    print("TEST 2: Predict Endpoint - FAKE Job")
    print("=" * 60)
    fake_job = {
        "text": "URGENT HIRING!! Work From Home!! Earn 100000 per week easily!! No experience needed!! HURRY only 5 seats!! Registration fee 499 refundable!! WhatsApp or email",
        "title": "Work From Home",
        "company": "Unknown"
    }

    try:
        response = requests.post(f"{BASE_URL}/predict", json=fake_job)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Prediction successful")
            print(f"   Verdict: {result['prediction']}")
            print(f"   Confidence: {result['confidence']}%")
            print(f"   Risk Level: {result['risk_level']}")
            print(f"   Risk Flags: {result['risk_flags']}")
            return True
        else:
            print(f"❌ Error: Status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_real_job():
    """Test prediction with real job"""
    print("\n" + "=" * 60)
    print("TEST 3: Predict Endpoint - REAL Job")
    print("=" * 60)
    real_job = {
        "text": "Infosys is hiring a Senior Software Engineer for our Cloud Platform team. Requirements: 4+ years Python/Go, AWS/Kubernetes. B.Tech in CS. CTC: 18-28 LPA. Apply: careers.infosys.com",
        "title": "Senior Software Engineer",
        "company": "Infosys"
    }

    try:
        response = requests.post(f"{BASE_URL}/predict", json=real_job)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Prediction successful")
            print(f"   Verdict: {result['prediction']}")
            print(f"   Confidence: {result['confidence']}%")
            print(f"   Risk Level: {result['risk_level']}")
            return True
        else:
            print(f"❌ Error: Status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_batch():
    """Test batch prediction"""
    print("\n" + "=" * 60)
    print("TEST 4: Batch Predict Endpoint")
    print("=" * 60)
    batch_jobs = {
        "jobs": [
            {
                "text": "URGENT!! Easy Money!! No skills needed!!",
                "title": "Easy Work",
                "company": "Unknown"
            },
            {
                "text": "Google is hiring a Software Engineer. Requirements: CS degree, strong algorithms. Location: Mountain View. Apply: google.com/careers",
                "title": "Software Engineer",
                "company": "Google"
            }
        ]
    }

    try:
        response = requests.post(f"{BASE_URL}/predict/batch", json=batch_jobs)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Batch prediction successful")
            print(f"   Total analyzed: {result['total']}")
            print(f"   Results:")
            for i, res in enumerate(result['results'], 1):
                print(f"     Job {i}: {res['prediction']} ({res['confidence']}%)")
            return True
        else:
            print(f"❌ Error: Status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    results = []
    
    results.append(("Health Check", test_health()))
    time.sleep(1)
    results.append(("Fake Job Prediction", test_fake_job()))
    time.sleep(1)
    results.append(("Real Job Prediction", test_real_job()))
    time.sleep(1)
    results.append(("Batch Prediction", test_batch()))
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 60)
