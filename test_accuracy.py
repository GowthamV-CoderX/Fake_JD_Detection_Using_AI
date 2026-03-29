#!/usr/bin/env python3
"""
Test improved fake job detection
"""

from feature_extractor import FeatureExtractor
from rule_engine import RuleEngine
from hybrid_detector import HybridDetector

def test_fake_job():
    # Test the fake job
    fake_job_text = 'URGENT HIRING!! Work From Home!! Earn 100000 per week easily!! No experience needed!! HURRY only 5 seats!! Registration fee 499 refundable!! WhatsApp or email for details'
    fake_title = 'Easy Money'
    fake_company = 'Unknown'

    print('🔍 TESTING IMPROVED FAKE JOB DETECTION')
    print('=' * 60)
    print(f'Job: {fake_job_text[:80]}...')
    print()

    # Extract features
    extractor = FeatureExtractor()
    features = extractor.extract(fake_job_text, fake_title, fake_company)

    print('📊 EXTRACTED FEATURES:')
    print(f'  Word count: {features.word_count}')
    print(f'  Exclamation count: {features.exclamation_count}')
    print(f'  Urgency phrases: {features.urgency_phrase_count}')
    print(f'  Vague promises: {features.vague_promise_count}')
    print(f'  Payment requests: {features.payment_request_count}')
    print(f'  WhatsApp mentions: {features.whatsapp_mentions}')
    print(f'  Has unrealistic salary: {features.has_unrealistic_salary}')
    print(f'  Has generic email: {features.has_generic_email}')
    print(f'  Spam signal score: {features.spam_signal_score}')
    print(f'  Professionalism score: {features.professionalism_score}')
    print()

    # Check rules
    rule_engine = RuleEngine()
    rule_output = rule_engine.evaluate(features, fake_job_text)

    print('🚩 RULE ENGINE RESULTS:')
    print(f'  Rule score: {rule_output.rule_score}')
    print(f'  Risk level: {rule_output.risk_level}')
    print(f'  Active rules: {len(rule_output.active_rules)}')
    for rule in rule_output.active_rules[:5]:  # Show first 5
        print(f'    - {rule.flag_message} (severity: {rule.severity})')
    if len(rule_output.active_rules) > 5:
        print(f'    ... and {len(rule_output.active_rules) - 5} more')
    print()

    # Test hybrid detector
    detector = HybridDetector(use_embeddings=False)
    result = detector.predict(fake_job_text, fake_title, fake_company)

    print('🤖 HYBRID DETECTOR RESULT:')
    print(f'  Prediction: {result.prediction}')
    print(f'  Confidence: {result.confidence}%')
    print(f'  Risk Level: {result.risk_level}')
    print(f'  Composite score: {result.scores["composite"]:.3f}')
    print(f'  Rule engine score: {result.scores["rule_engine"]:.3f}')
    print(f'  Spam signal score: {result.scores["spam_signal"]:.3f}')

    print()
    print('🎯 ACCURACY CHECK:')
    if result.prediction == 'FAKE':
        print('✅ SUCCESS: Fake job correctly identified as FAKE!')
        print('   Improved accuracy working!')
    else:
        print('❌ FAILED: Fake job still classified as REAL')
        print('   Need further improvements')

    return result.prediction == 'FAKE'

def test_real_job():
    # Test a real job
    real_job_text = 'Software Engineer position at Google. 5+ years experience required. Competitive salary, benefits, and remote work options. Python, Java, and cloud experience preferred.'
    real_title = 'Software Engineer'
    real_company = 'Google'

    print('\n' + '=' * 60)
    print('🔍 TESTING REAL JOB DETECTION')
    print('=' * 60)
    print(f'Job: {real_job_text[:80]}...')
    print()

    detector = HybridDetector(use_embeddings=False)
    result = detector.predict(real_job_text, real_title, real_company)

    print('🤖 HYBRID DETECTOR RESULT:')
    print(f'  Prediction: {result.prediction}')
    print(f'  Confidence: {result.confidence}%')
    print(f'  Risk Level: {result.risk_level}')

    if result.prediction == 'REAL':
        print('✅ SUCCESS: Real job correctly identified as REAL!')
    else:
        print('❌ FAILED: Real job incorrectly classified as FAKE')

    return result.prediction == 'REAL'

if __name__ == '__main__':
    fake_correct = test_fake_job()
    real_correct = test_real_job()

    print('\n' + '=' * 60)
    print('📊 FINAL ACCURACY RESULTS')
    print('=' * 60)
    print(f'Fake job detection: {"✅ PASS" if fake_correct else "❌ FAIL"}')
    print(f'Real job detection: {"✅ PASS" if real_correct else "❌ FAIL"}')
    print(f'Overall accuracy: {100 * (fake_correct + real_correct) / 2:.0f}%')

    if fake_correct and real_correct:
        print('\n🎉 EXCELLENT! Both fake and real jobs correctly classified!')
        print('🚀 Project accuracy significantly improved!')
    elif fake_correct:
        print('\n👍 Good progress! Fake jobs now detected correctly.')
        print('   Real job detection may need minor tuning.')
    else:
        print('\n⚠️  Fake job detection still needs work.')
        print('   Further improvements required.')