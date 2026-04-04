"""
AI-Generated Text Detector
Uses lightweight statistical/linguistic heuristics to detect ChatGPT/Gemini/Claude-written text.
Zero external dependencies beyond what's already in the project.
"""

import re
import math
from collections import Counter


# ─── Phrase Banks ────────────────────────────────────────────────────────────

AI_FILLER_PHRASES = [
    # ChatGPT / LLM signature openers
    r"\bwe are looking for\b",
    r"\bwe are seeking\b",
    r"\bjoin our dynamic team\b",
    r"\bfast-paced environment\b",
    r"\bpassionate (about|individual|professional)\b",
    r"\bteam player\b",
    r"\bself-starter\b",
    r"\bgo-getter\b",
    r"\bexcellent communication skills\b",
    r"\bstrong attention to detail\b",
    r"\bstrong analytical skills\b",
    r"\bproven track record\b",
    r"\bresults-driven\b",
    r"\bdata-driven\b",
    r"\bthought leader\b",
    r"\bsynergize?\b",
    r"\bleverage\b",
    r"\bpivot\b",
    r"\bscalable solution\b",
    r"\bholistic approach\b",
    r"\bseamlessly\b",
    r"\brobust (experience|background|skill)\b",
    r"\bin a collaborative\b",
    r"\bcross-functional\b",
    r"\bin this role\b",
    r"\bkey responsibilities (include|are)\b",
    r"\bresponsibilities include but are not limited to\b",
    r"\bwe offer a competitive\b",
    r"\bequal opportunity employer\b",
    r"\bwork life balance\b",
    r"\bwork-life balance\b",
    r"\bgrowth mindset\b",
    r"\boutside the box\b",
    r"\bmove the needle\b",
    r"\bbring to the table\b",
    r"\bvalue-add\b",
    r"\bstakeholder(s)?\b",
    r"\bend-to-end\b",
    r"\bas a (key|valued) member\b",
    r"\bbest-in-class\b",
    r"\bworld-class\b",
    r"\bexciting opportunity\b",
    r"\bunique opportunity\b",
    r"\brapidly growing\b",
    r"\bhigh-growth\b",
    r"\bsolution-oriented\b",
    r"\bcritical thinking\b",
    r"\bproblem-solving skills\b",
    r"\btime management skills\b",
    r"\borganizational skills\b",
]

AI_STRUCTURE_PATTERNS = [
    # Bullet-heavy structured sections typical of LLM output
    r"(responsibilities|requirements|qualifications|skills|what you.ll do|what we offer)",
    r"(minimum qualifications|preferred qualifications|nice to have)",
    r"(about the role|about us|about the company|about the position)",
    r"(what you.ll bring|what you bring|what you will bring)",
]

HUMAN_SCAM_TELLS = [
    r"\bno experience (needed|required)\b",
    r"\bwork from home\b.*\bearning\b",
    r"\beasy money\b",
    r"\bmake \$\d+",
    r"\bguaranteed income\b",
    r"\buncapped earning\b",
    r"whatsapp.*apply",
    r"apply.*whatsapp",
    r"\btelegram\b",
    r"\bgoogle form\b",
    r"\bfee.*registration\b",
    r"\bregistration.*fee\b",
]


# ─── Linguistic Feature Extractors ───────────────────────────────────────────

def _sentences(text: str) -> list[str]:
    """Split text into sentences."""
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 10]


def _words(text: str) -> list[str]:
    return re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())


def _avg_sentence_length(sentences: list[str]) -> float:
    if not sentences:
        return 0
    lengths = [len(s.split()) for s in sentences]
    return sum(lengths) / len(lengths)


def _sentence_length_variance(sentences: list[str]) -> float:
    """Low variance = uniform = AI-like. High variance = human-like."""
    if len(sentences) < 2:
        return 100.0
    lengths = [len(s.split()) for s in sentences]
    mean = sum(lengths) / len(lengths)
    variance = sum((l - mean) ** 2 for l in lengths) / len(lengths)
    return variance


def _type_token_ratio(words: list[str]) -> float:
    """Vocabulary diversity. AI tends to be moderate — not too repetitive, not too rich."""
    if not words:
        return 0
    return len(set(words)) / len(words)


def _count_ai_phrases(text: str) -> int:
    text_lower = text.lower()
    count = 0
    for pattern in AI_FILLER_PHRASES:
        if re.search(pattern, text_lower):
            count += 1
    return count


def _count_structure_sections(text: str) -> int:
    text_lower = text.lower()
    count = 0
    for pattern in AI_STRUCTURE_PATTERNS:
        if re.search(pattern, text_lower):
            count += 1
    return count


def _bullet_density(text: str) -> float:
    """Ratio of bullet/list lines to total lines."""
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if not lines:
        return 0
    bullet_lines = sum(1 for l in lines if re.match(r'^[\-\*\•\✓\→\►]|^\d+[\.\)]', l))
    return bullet_lines / len(lines)


def _has_perfect_grammar_structure(sentences: list[str]) -> bool:
    """Check for unnaturally consistent capitalization + punctuation = AI tell."""
    if len(sentences) < 3:
        return False
    well_formed = sum(
        1 for s in sentences
        if s and s[0].isupper() and s[-1] in '.!?'
    )
    return (well_formed / len(sentences)) > 0.85


def _count_human_scam_tells(text: str) -> int:
    text_lower = text.lower()
    return sum(1 for p in HUMAN_SCAM_TELLS if re.search(p, text_lower))


# ─── Main Scoring Function ────────────────────────────────────────────────────

def compute_ai_score(text: str) -> dict:
    """
    Returns a dict with:
      - ai_score: 0.0–1.0 (probability of AI generation)
      - signals: list of triggered signals for explainability
      - verdict: 'ai_generated' | 'human_written'
    """
    sentences = _sentences(text)
    words = _words(text)

    signals = []
    score = 0.0

    # 1. AI filler phrase count (weight: heavy)
    ai_phrase_count = _count_ai_phrases(text)
    if ai_phrase_count >= 8:
        score += 0.35
        signals.append(f"High AI filler phrases ({ai_phrase_count} matched)")
    elif ai_phrase_count >= 4:
        score += 0.20
        signals.append(f"Moderate AI filler phrases ({ai_phrase_count} matched)")
    elif ai_phrase_count >= 2:
        score += 0.08
        signals.append(f"Some AI filler phrases ({ai_phrase_count} matched)")

    # 2. Structured section headers (weight: medium)
    section_count = _count_structure_sections(text)
    if section_count >= 3:
        score += 0.15
        signals.append(f"LLM-style section headers ({section_count} found)")
    elif section_count >= 2:
        score += 0.08

    # 3. Bullet density (weight: light)
    bd = _bullet_density(text)
    if bd > 0.45:
        score += 0.12
        signals.append(f"High bullet density ({bd:.0%})")
    elif bd > 0.25:
        score += 0.05

    # 4. Sentence length variance — low variance = AI
    if len(sentences) >= 4:
        variance = _sentence_length_variance(sentences)
        if variance < 15:
            score += 0.15
            signals.append("Very uniform sentence lengths (AI-like)")
        elif variance < 30:
            score += 0.07
            signals.append("Moderately uniform sentences")

    # 5. Perfect grammar structure
    if _has_perfect_grammar_structure(sentences):
        score += 0.10
        signals.append("Unnaturally consistent grammar/capitalization")

    # 6. Type-token ratio — AI is moderate (0.45–0.65)
    ttr = _type_token_ratio(words)
    if 0.40 <= ttr <= 0.68 and len(words) > 50:
        score += 0.05
        signals.append(f"Moderate vocabulary diversity (TTR={ttr:.2f})")

    # 7. Average sentence length — AI prefers 15–25 words
    avg_len = _avg_sentence_length(sentences)
    if 14 <= avg_len <= 26:
        score += 0.05
        signals.append(f"AI-typical avg sentence length ({avg_len:.1f} words)")

    # 8. Human scam tells REDUCE the AI score
    scam_tells = _count_human_scam_tells(text)
    if scam_tells >= 2:
        score -= 0.25
        signals.append(f"Human scam signals found ({scam_tells}) — less likely AI-generated")
    elif scam_tells == 1:
        score -= 0.10

    # Clamp
    score = max(0.0, min(1.0, score))

    verdict = "ai_generated" if score >= 0.45 else "human_written"

    return {
        "ai_score": round(score, 3),
        "signals": signals,
        "verdict": verdict,
        "ai_phrase_count": ai_phrase_count,
        "bullet_density": round(bd, 3),
        "section_count": section_count,
    }