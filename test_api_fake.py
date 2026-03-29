#!/usr/bin/env python3
"""
Test fake job detection via API
"""

import requests
import json
import time

def test_fake_job_api():
    # Wait for server to start
    time.sleep(3)

    # Test fake job through API
    fake_job_data = {
        'text': 'URGENT HIRING!! Work From Home!! Earn 100000 per week easily!! No experience needed!! HURRY only 5 seats!! Registration fee 499 refundable!! WhatsApp or email for details',
        'title': 'Easy Money',
        'company': 'Unknown'
    }

    try:
        response = requests.post('http://localhost:8000/predict', json=fake_job_data, timeout=10)
        if response.status_code == 200:
            result = response.json()
            print('Prediction - FAKE Job (via API)')
            print('-' * 70)
            print(f'Verdict: {result["prediction"]}')
            print(f'Confidence: {result["confidence"]}%')
            print(f'Risk Level: {result["risk_level"]}')
            print(f'Risk Flags: {len(result["risk_flags"])}')
            for flag in result['risk_flags']:
                print(f'  - {flag}')
        else:
            print(f'API Error: {response.status_code} - {response.text}')
    except Exception as e:
        print(f'Error: {e}')

if __name__ == "__main__":
    test_fake_job_api()