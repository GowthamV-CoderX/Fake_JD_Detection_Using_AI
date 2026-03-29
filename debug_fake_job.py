#!/usr/bin/env python3
"""
Debug script to analyze why fake job detection is failing
"""

from feature_extractor import FeatureExtractor
from rule_engine import RuleEngine
from hybrid_detector import HybridDetector

def main():
    # Test the fake job
    fake_job_text = 'URGENT HIRING!! Work From Home!! Earn 100000 per week easily!! No experience needed!! HURRY only 5 seats!! Registration fee 499 refundable!! WhatsApp or email for details'
    fake_title = 'Easy Money'
    fake_company = 'Unknown'

    print('🔍 ANALYZING FAKE JOB FEATURES & RULES')
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
    for rule in rule_output.active_rules:
        print(f'    - {rule.flag_message} (severity: {rule.severity})')
    print()

    # Test hybrid detector
    detector = HybridDetector(use_embeddings=False)
    result = detector.predict(fake_job_text, fake_title, fake_company)

    print('🤖 HYBRID DETECTOR RESULT:')
    print(f'  Prediction: {result.prediction}')
    print(f'  Confidence: {result.confidence}%')
    print(f'  Risk Level: {result.risk_level}')
    print(f'  Composite score: {result.scores["composite"]}')
    print(f'  Rule engine score: {result.scores["rule_engine"]}')
    print(f'  Spam signal score: {result.scores["spam_signal"]}')

    print()
    print('🔧 IMPROVEMENT NEEDED:')
    if result.prediction == 'REAL':
        print('❌ FAKE job incorrectly classified as REAL')
        print('   Need to lower threshold or increase rule weights')
    else:
        print('✅ FAKE job correctly classified')

if __name__ == '__main__':
    main()