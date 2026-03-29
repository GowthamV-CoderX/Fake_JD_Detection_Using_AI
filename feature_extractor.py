"""
src/features/feature_extractor.py
Extracts structured features from job description text for ML models.
"""

import re
import math
from dataclasses import dataclass, field
from typing import List, Optional


# ──────────────────────────────────────────────
# PATTERNS
# ──────────────────────────────────────────────
GENERIC_EMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
    "rediffmail.com", "ymail.com", "aol.com", "mail.com",
}

SALARY_PATTERN = re.compile(
    r"(₹|rs\.?|inr|lpa|per\s+week|per\s+month|/month|/week|lakh|crore|\d[\d,]*\s*k)",
    re.IGNORECASE,
)

UNREALISTIC_SALARY_PATTERN = re.compile(
    r"(earn\s+₹[\d,]+\s*(per|/)\s*week|₹\s*[\d,]+\s*lakh\s*per\s*(week|day|month)|"
    r"\d+\s*lakh.*easily|guaranteed.*income|unlimited.*earning|no\s+upper\s+limit|"
    r"easy\s+money|quick\s+money|fast\s+money|make\s+₹[\d,]+\s*per\s*week)",
    re.IGNORECASE,
)

EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
)

PHONE_PATTERN = re.compile(
    r"(\+91[\-\s]?)?\d[\dXx]{9}"  # Indian phone numbers, sometimes masked
)

URGENCY_PATTERN = re.compile(
    r"\b(urgent|hurry|act\s+now|limited\s+seats?|today\s+only|closing\s+(tonight|today)|"
    r"don['']t\s+miss|last\s+chance|only\s+\d+\s+seats?|immediate\s+joining|"
    r"apply\s+now|quick\s+hiring|fast\s+track|rush|deadline|expires?\s+soon)\b",
    re.IGNORECASE,
)

PAYMENT_REQUEST_PATTERN = re.compile(
    r"\b(registration\s+fee|security\s+deposit|starter\s+kit|refundable|"
    r"pay\s+₹|deposit\s+₹|fees?\s+required|investment\s+required|"
    r"pay\s+to\s+join|joining\s+fee)\b",
    re.IGNORECASE,
)

COMPANY_DETAILS_PATTERN = re.compile(
    r"\b(pvt\.?\s*ltd\.?|limited|inc\.?|corporation|llp|technologies|solutions|"
    r"careers?\.|\.com|\.in|\.org)\b",
    re.IGNORECASE,
)

VAGUE_PROMISE_PATTERN = re.compile(
    r"\b(anyone\s+can|no\s+experience\s+(needed|required)|no\s+qualification|"
    r"work\s+from\s+anywhere|guaranteed\s+(pay|salary|income|earnings?)|"
    r"100%\s+guaranteed|easy\s+money|no\s+skills?\s+needed|beginner\s+friendly|"
    r"passive\s+income|be\s+your\s+own\s+boss|financial\s+freedom|"
    r"extra\s+income|side\s+(hustle|income)|home\s+based|part\s+time|"
    r"flexible\s+hours|set\s+your\s+own\s+schedule|work\s+from\s+home)\b",
    re.IGNORECASE,
)

EXCLAMATION_PATTERN = re.compile(r"!+")
EMOJI_PATTERN = re.compile(
    "[\U00010000-\U0010ffff]|[\U0001F300-\U0001F9FF]|[\u2600-\u26FF]|[\u2700-\u27BF]",
    flags=re.UNICODE,
)
CAPS_WORD_PATTERN = re.compile(r"\b[A-Z]{3,}\b")  # fully capitalised words ≥3 chars

NETWORK_MARKETING_PATTERN = re.compile(
    r"\b(network\s+marketing|mlm|multi.?level|recruit|downline|referral\s+bonus|"
    r"bring\s+others|pyramid)\b",
    re.IGNORECASE,
)

WHATSAPP_PATTERN = re.compile(r"\bwhatsapp\b", re.IGNORECASE)


# ──────────────────────────────────────────────
# DATACLASS RESULT
# ──────────────────────────────────────────────
@dataclass
class ExtractedFeatures:
    # Text stats
    word_count: int = 0
    char_count: int = 0
    sentence_count: int = 0
    avg_word_length: float = 0.0
    avg_sentence_length: float = 0.0

    # Salary features
    has_salary_mention: bool = False
    has_unrealistic_salary: bool = False
    salary_mentions_count: int = 0

    # Contact features
    email_count: int = 0
    phone_count: int = 0
    has_generic_email: bool = False
    has_company_email: bool = False
    contact_emails: List[str] = field(default_factory=list)

    # Red-flag linguistic features
    exclamation_count: int = 0
    emoji_count: int = 0
    caps_word_count: int = 0
    urgency_phrase_count: int = 0
    vague_promise_count: int = 0
    payment_request_count: int = 0
    network_marketing_signals: int = 0
    whatsapp_mentions: int = 0

    # Company signals
    has_company_details: bool = False
    has_domain_mention: bool = False

    # Computed scores
    spam_signal_score: float = 0.0     # 0–1, high = more spam-like
    professionalism_score: float = 0.0  # 0–1, high = more professional


# ──────────────────────────────────────────────
# EXTRACTOR
# ──────────────────────────────────────────────
class FeatureExtractor:
    """Extracts ~20 structured features from a job description string."""

    def extract(self, text: str, title: str = "", company: str = "") -> ExtractedFeatures:
        full_text = f"{title} {company} {text}".strip()
        words = full_text.split()
        sentences = re.split(r"[.!?]+", full_text)
        sentences = [s.strip() for s in sentences if s.strip()]

        feat = ExtractedFeatures()

        # ── Basic stats ──
        feat.word_count = len(words)
        feat.char_count = len(full_text)
        feat.sentence_count = max(len(sentences), 1)
        feat.avg_word_length = (
            sum(len(w) for w in words) / len(words) if words else 0
        )
        feat.avg_sentence_length = feat.word_count / feat.sentence_count

        # ── Salary ──
        salary_matches = SALARY_PATTERN.findall(full_text)
        feat.salary_mentions_count = len(salary_matches)
        feat.has_salary_mention = feat.salary_mentions_count > 0
        feat.has_unrealistic_salary = bool(UNREALISTIC_SALARY_PATTERN.search(full_text))

        # ── Contacts ──
        emails = EMAIL_PATTERN.findall(full_text)
        feat.email_count = len(emails)
        feat.contact_emails = emails

        for email in emails:
            domain = email.split("@")[-1].lower()
            if domain in GENERIC_EMAIL_DOMAINS:
                feat.has_generic_email = True
            else:
                feat.has_company_email = True

        phones = PHONE_PATTERN.findall(full_text)
        feat.phone_count = len(phones)

        # ── Red flags ──
        feat.exclamation_count = len(EXCLAMATION_PATTERN.findall(full_text))
        feat.emoji_count = len(EMOJI_PATTERN.findall(full_text))
        feat.caps_word_count = len(CAPS_WORD_PATTERN.findall(full_text))
        feat.urgency_phrase_count = len(URGENCY_PATTERN.findall(full_text))
        feat.vague_promise_count = len(VAGUE_PROMISE_PATTERN.findall(full_text))
        feat.payment_request_count = len(PAYMENT_REQUEST_PATTERN.findall(full_text))
        feat.network_marketing_signals = len(NETWORK_MARKETING_PATTERN.findall(full_text))
        feat.whatsapp_mentions = len(WHATSAPP_PATTERN.findall(full_text))

        # ── Company ──
        feat.has_company_details = bool(COMPANY_DETAILS_PATTERN.search(full_text))
        feat.has_domain_mention = bool(re.search(r"\.\s*(com|in|org|net|co)\b", full_text, re.IGNORECASE))

        # ── Computed scores ──
        feat.spam_signal_score = self._compute_spam_score(feat)
        feat.professionalism_score = self._compute_professionalism_score(feat)

        return feat

    def _compute_spam_score(self, f: ExtractedFeatures) -> float:
        """Normalised spam indicator 0–1."""
        score = 0.0
        score += min(f.exclamation_count / 4.0, 1.0) * 0.18  # Increased weight
        score += min(f.emoji_count / 3.0, 1.0) * 0.12        # Increased weight
        score += min(f.caps_word_count / 8.0, 1.0) * 0.12    # Increased weight
        score += min(f.urgency_phrase_count / 2.0, 1.0) * 0.20  # Increased weight
        score += min(f.vague_promise_count / 2.0, 1.0) * 0.18  # Increased weight
        score += min(f.payment_request_count / 1.0, 1.0) * 0.25  # Major increase
        score += (1.0 if f.has_unrealistic_salary else 0.0) * 0.15  # Increased
        score += (1.0 if f.has_generic_email and not f.has_company_email else 0.0) * 0.12
        score += min(f.network_marketing_signals / 1.0, 1.0) * 0.08
        score += (1.0 if f.whatsapp_mentions > 0 else 0.0) * 0.10  # New weight for WhatsApp
        return round(min(score, 1.0), 4)

    def _compute_professionalism_score(self, f: ExtractedFeatures) -> float:
        """Normalised professionalism indicator 0–1."""
        score = 0.0
        # Reasonable word count (150–800 words)
        if 150 <= f.word_count <= 800:
            score += 0.20
        elif 80 <= f.word_count < 150 or 800 < f.word_count <= 1200:
            score += 0.10
        score += 0.15 if f.has_company_details else 0.0
        score += 0.15 if f.has_company_email else 0.0
        score += 0.15 if f.has_domain_mention else 0.0
        score += 0.10 if f.has_salary_mention and not f.has_unrealistic_salary else 0.0
        # Penalise chaos
        score -= min(f.exclamation_count / 10.0, 0.15)
        score -= min(f.emoji_count / 10.0, 0.10)
        return round(max(0.0, min(score, 1.0)), 4)

    def to_vector(self, feat: ExtractedFeatures) -> List[float]:
        """Return a flat numerical vector for ML consumption."""
        return [
            feat.word_count,
            feat.char_count,
            feat.sentence_count,
            feat.avg_word_length,
            feat.avg_sentence_length,
            float(feat.has_salary_mention),
            float(feat.has_unrealistic_salary),
            feat.salary_mentions_count,
            feat.email_count,
            feat.phone_count,
            float(feat.has_generic_email),
            float(feat.has_company_email),
            feat.exclamation_count,
            feat.emoji_count,
            feat.caps_word_count,
            feat.urgency_phrase_count,
            feat.vague_promise_count,
            feat.payment_request_count,
            feat.network_marketing_signals,
            feat.whatsapp_mentions,
            float(feat.has_company_details),
            float(feat.has_domain_mention),
            feat.spam_signal_score,
            feat.professionalism_score,
        ]

    @staticmethod
    def feature_names() -> List[str]:
        return [
            "word_count", "char_count", "sentence_count",
            "avg_word_length", "avg_sentence_length",
            "has_salary_mention", "has_unrealistic_salary", "salary_mentions_count",
            "email_count", "phone_count", "has_generic_email", "has_company_email",
            "exclamation_count", "emoji_count", "caps_word_count",
            "urgency_phrase_count", "vague_promise_count", "payment_request_count",
            "network_marketing_signals", "whatsapp_mentions",
            "has_company_details", "has_domain_mention",
            "spam_signal_score", "professionalism_score",
        ]
