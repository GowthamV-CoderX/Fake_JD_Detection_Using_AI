"""
rule_engine.py  —  Enhanced accuracy version
Location: rule_engine.py (repo root)

IMPROVEMENTS vs original:
  - 25 rules (was 11) covering more scam patterns
  - Better combo detection with more signal combinations
  - Indian scam patterns specifically added
  - Severity scores tuned based on real scam data
  - Human-readable flag descriptions for frontend display
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import re


@dataclass
class Rule:
    name: str
    severity: float
    pattern: str
    description: str  # shown to user in UI


@dataclass
class RuleEngineOutput:
    rule_score: float
    active_rules: List[Rule] = field(default_factory=list)
    triggered_flags: List[str] = field(default_factory=list)
    flag_descriptions: List[str] = field(default_factory=list)


class RuleEngine:
    def __init__(self):
        self.rules: List[Rule] = [

            # ── CRITICAL signals (severity 0.9–1.0) ──────────────────────────
            Rule("payment_required", 1.0,
                 r"(registration\s+fee|pay\s+to\s+join|joining\s+fee|deposit\s+required"
                 r"|security\s+deposit|starter\s+kit\s+fee|refundable\s+deposit"
                 r"|pay\s+₹|fees?\s+required|investment\s+required)",
                 "Asks applicant to pay a fee or deposit"),

            Rule("unrealistic_salary", 0.95,
                 r"(earn\s+₹[\d,]+\s*(per|/)\s*week|₹\s*[\d,]+\s*lakh\s*per\s*(week|day)"
                 r"|\d+\s*lakh.*easily|guaranteed.*income|unlimited.*earning"
                 r"|no\s+upper\s+limit|easy\s+money|₹?\d{5,}\s*per\s*day"
                 r"|make\s+₹[\d,]+\s*per\s*week|earn\s+upto\s+₹)",
                 "Unrealistic / guaranteed salary claim"),

            Rule("mlm_network", 0.90,
                 r"(network\s+marketing|mlm|multi.?level\s+marketing"
                 r"|recruit\s+others|downline|referral\s+bonus\s+per\s+join"
                 r"|bring\s+(friends|others|members)|pyramid\s+scheme"
                 r"|chain\s+marketing|direct\s+selling\s+plan)",
                 "Network marketing / MLM / pyramid scheme detected"),

            Rule("fee_refund_trap", 0.90,
                 r"(refundable|will\s+be\s+returned|money\s+back\s+guarantee"
                 r"|get\s+your\s+money\s+back|deposit\s+refunded\s+after"
                 r"|fee\s+is\s+refundable)",
                 "Classic fee-then-refund trap pattern"),

            # ── HIGH signals (severity 0.7–0.85) ─────────────────────────────
            Rule("generic_email_only", 0.85,
                 r"\b([\w._%+\-]+@(gmail|yahoo|hotmail|outlook|rediffmail|ymail|aol)\.com)\b",
                 "Uses generic email (Gmail/Yahoo) instead of company email"),

            Rule("whatsapp_contact", 0.80,
                 r"\b(whatsapp|watsapp|wa\.me|whats\s*app)\b",
                 "Contact via WhatsApp only — not professional"),

            Rule("urgency_pressure", 0.75,
                 r"\b(urgent|hurry|act\s+now|limited\s+seats?|today\s+only"
                 r"|last\s+chance|only\s+\d+\s+(seats?|spots?|openings?)"
                 r"|immediate\s+joining|apply\s+before|closing\s+tonight"
                 r"|don['\u2019]t\s+miss|expires?\s+soon|rush|deadline\s+today)\b",
                 "High-pressure urgency tactics used"),

            Rule("no_experience_needed", 0.75,
                 r"\b(no\s+experience\s+(needed|required|necessary)"
                 r"|zero\s+experience|freshers?\s+only|anyone\s+can\s+(do|apply|join)"
                 r"|no\s+qualification\s+needed|8th\s+pass|10th\s+pass\s+eligible"
                 r"|no\s+skills?\s+(needed|required))\b",
                 "No skills or experience required — unusually low barrier"),

            Rule("vague_promises", 0.70,
                 r"\b(be\s+your\s+own\s+boss|financial\s+freedom|passive\s+income"
                 r"|extra\s+income|side\s+(hustle|income|earnings?)"
                 r"|work\s+from\s+anywhere|set\s+your\s+own\s+hours"
                 r"|unlimited\s+potential|sky\s+is\s+the\s+limit"
                 r"|change\s+your\s+life|dream\s+(job|income|lifestyle))\b",
                 "Vague lifestyle / passive income promises"),

            Rule("too_good_claims", 0.70,
                 r"\b(guaranteed\s+(pay|salary|income|earnings?|job)"
                 r"|100%\s+guaranteed|assured\s+(income|salary|job)"
                 r"|never\s+lose\s+money|zero\s+risk|risk.?free\s+income)\b",
                 "Too-good-to-be-true guaranteed income claims"),

            # ── MEDIUM signals (severity 0.5–0.65) ───────────────────────────
            Rule("masked_phone", 0.65,
                 r"\b\d{2,4}[xX*]{4,}\d{0,4}\b",
                 "Phone number partially masked — suspicious"),

            Rule("affiliate_scam", 0.65,
                 r"\b(earn\s+per\s+click|per\s+referral|per\s+signup"
                 r"|affiliate\s+earn|cpa\s+offer|earn\s+per\s+download"
                 r"|data\s+entry\s+earn|per\s+task\s+₹|captcha\s+entry"
                 r"|ad\s+posting\s+job|copy\s+paste\s+job)\b",
                 "Affiliate / data-entry / per-click scam pattern"),

            Rule("no_company_info", 0.60,
                 r"^(?!.*(pvt|ltd|limited|inc|corp|technologies|solutions"
                 r"|enterprises|industries|group|associates)).*$",
                 "No company name or registration details found"),

            Rule("excessive_exclamations", 0.55,
                 r"(!{2,}|(?:.*!){5,})",
                 "Excessive exclamation marks — spam-like tone"),

            Rule("all_caps_abuse", 0.50,
                 r"(([A-Z]{4,}\s+){3,})",
                 "Excessive ALL CAPS usage — unprofessional"),

            Rule("emoji_overload", 0.50,
                 r"([\U0001F300-\U0001F9FF].*){4,}",
                 "Too many emojis — not a professional job post"),

            Rule("work_from_home_vague", 0.50,
                 r"\b(work\s+from\s+home|home\s+based\s+job|online\s+job"
                 r"|laptop\s+job|mobile\s+se\s+kaam|ghar\s+baithe)\b",
                 "Generic work-from-home claim with no details"),

            Rule("no_job_structure", 0.55,
                 r"^(?!.*(responsibilities|requirements|qualifications"
                 r"|duties|skills\s+required|job\s+description"
                 r"|what\s+you.ll\s+do|role\s+overview)).*$",
                 "Missing standard job structure (responsibilities/requirements)"),

            Rule("suspicious_word_count", 0.45,
                 r"^.{0,200}$",
                 "Very short description — real jobs have detailed descriptions"),

            Rule("telegram_contact", 0.70,
                 r"\b(telegram|t\.me/|@\w+\s+telegram)\b",
                 "Contact via Telegram only — not professional"),

            Rule("google_form_apply", 0.60,
                 r"\b(google\s+form|bit\.ly|tinyurl|forms\.gle"
                 r"|apply\s+via\s+link|click\s+the\s+link\s+to\s+apply)\b",
                 "Application via anonymous Google Form / short link"),

            Rule("hindi_scam_phrases", 0.75,
                 r"\b(ghar\s+baithe|paisa\s+kamao|daily\s+income"
                 r"|roz\s+kamai|free\s+registration|instant\s+payout"
                 r"|paise\s+kamao|online\s+paise|daily\s+earning)\b",
                 "Hindi scam phrases detected"),

            Rule("data_entry_scam", 0.65,
                 r"\b(data\s+entry\s+operator|form\s+filling\s+job"
                 r"|typing\s+job\s+at\s+home|copy\s+paste\s+work"
                 r"|simple\s+typing|online\s+typing)\b",
                 "Data entry / typing job scam pattern"),

            Rule("investment_scheme", 0.85,
                 r"\b(invest\s+₹[\d,]+|investment\s+plan|returns?\s+on\s+investment"
                 r"|roi\s+guaranteed|double\s+your\s+money|triple\s+returns?"
                 r"|invest\s+and\s+earn|trading\s+income\s+guaranteed)\b",
                 "Investment scheme disguised as a job"),

            Rule("no_skills_mentioned", 0.40,
                 r"^(?!.*(python|java|sql|excel|aws|react|node|angular"
                 r"|machine\s+learning|data\s+analysis|accounting"
                 r"|marketing|sales|communication|management"
                 r"|design|writing|teaching|nursing|engineering)).*$",
                 "No specific skills mentioned for the role"),
        ]

    def evaluate(self, features, text: str) -> RuleEngineOutput:
        score = 0.0
        active_rules: List[Rule] = []
        triggered_flags: List[str] = []
        flag_descriptions: List[str] = []

        text_lower = text.lower()
        matched: List[str] = []

        for rule in self.rules:
            if re.search(rule.pattern, text_lower, re.DOTALL):
                score += rule.severity
                matched.append(rule.name)
                active_rules.append(rule)
                triggered_flags.append(rule.name)
                flag_descriptions.append(rule.description)

        # ── Combo bonuses ─────────────────────────────────────────────────────
        if "payment_required" in matched and "fee_refund_trap" in matched:
            score += 1.5
            triggered_flags.append("combo_fee_refund_trap")
            flag_descriptions.append("Fee + refund trap combo — classic scam pattern")

        if "generic_email_only" in matched and "urgency_pressure" in matched:
            score += 1.0
            triggered_flags.append("combo_urgency_generic_email")
            flag_descriptions.append("Urgency + generic email combo — pressure scam")

        if "mlm_network" in matched and "unrealistic_salary" in matched:
            score += 1.2
            triggered_flags.append("combo_mlm_salary")
            flag_descriptions.append("MLM + unrealistic salary — pyramid scheme")

        if "no_experience_needed" in matched and "unrealistic_salary" in matched:
            score += 1.0
            triggered_flags.append("combo_easy_money")
            flag_descriptions.append("No experience + high pay — too good to be true")

        if "whatsapp_contact" in matched and "payment_required" in matched:
            score += 1.0
            triggered_flags.append("combo_whatsapp_fee")
            flag_descriptions.append("WhatsApp contact + fee request — scam")

        if "data_entry_scam" in matched and "work_from_home_vague" in matched:
            score += 0.8
            triggered_flags.append("combo_data_entry_wfh")
            flag_descriptions.append("Data entry + WFH combo — common online scam")

        if "investment_scheme" in matched:
            score += 1.0
            triggered_flags.append("investment_job_disguise")
            flag_descriptions.append("Investment scheme disguised as employment")

        # Normalize to 0–1
        rule_score = min(score / 4.0, 1.0)

        return RuleEngineOutput(
            rule_score=rule_score,
            active_rules=active_rules,
            triggered_flags=triggered_flags,
            flag_descriptions=flag_descriptions,
        )