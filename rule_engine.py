"""
rule_engine.py
==============
Location in your repo: rule_engine.py  (root)

NO LOGIC CHANGES — this is your original file kept exactly as-is.
(No torch or sentence-transformers were ever used here.)
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
    description: str


@dataclass
class RuleEngineOutput:
    rule_score: float
    active_rules: List[Rule] = field(default_factory=list)
    triggered_flags: List[str] = field(default_factory=list)


class RuleEngine:
    def __init__(self):
        self.rules: List[Rule] = [
            # Hard scam signals
            Rule("payment_required", 1.0,
                 r"(fee|registration fee|pay.*join|deposit.*job)",
                 "Requests money"),
            Rule("external_contact", 0.95,
                 r"(whatsapp|telegram|gmail\.com|yahoo\.com|contact me)",
                 "External contact info"),
            Rule("urgency", 0.85,
                 r"(urgent|hurry|limited slots|only \d+ seats)",
                 "Creates urgency pressure"),
            Rule("high_salary", 0.9,
                 r"(₹?\d{4,}.*per day|\$\d{3,}.*per day)",
                 "Unrealistic salary"),

            # Medium signals
            Rule("no_experience", 0.75,
                 r"(no experience|required no experience|anyone can do)",
                 "Low barrier entry"),
            Rule("too_good", 0.7,
                 r"(easy money|guaranteed income|earn fast)",
                 "Too good to be true"),
            Rule("generic_remote", 0.65,
                 r"(work from home|remote job|online work)",
                 "Generic remote job"),

            # AI / fake structure
            Rule("ai_vague", 0.8,
                 r"(dynamic individual|fast-paced environment|team player)",
                 "AI-style vague text"),
            Rule("buzzwords", 0.7,
                 r"(innovative|cutting-edge|visionary|synergy)",
                 "Buzzword overload"),
            Rule("too_short", 0.75,
                 r"^.{0,120}$",
                 "Too short description"),
            Rule("no_structure", 0.7,
                 r"^(?!.*(requirements|responsibilities|qualifications)).*$",
                 "Missing job structure"),
        ]

    def evaluate(self, features, text: str) -> RuleEngineOutput:
        score = 0.0
        active_rules: List[Rule] = []
        triggered_flags: List[str] = []

        text_lower = text.lower()
        matched: List[str] = []

        for rule in self.rules:
            if re.search(rule.pattern, text_lower):
                score += rule.severity
                matched.append(rule.name)
                active_rules.append(rule)
                triggered_flags.append(rule.name)

        # Combo intelligence — extra score when multiple signals fire together
        if (
            "generic_remote" in matched
            and "no_experience" in matched
            and "high_salary" in matched
        ):
            score += 1.2
            triggered_flags.append("combo_remote_highpay")

        if "external_contact" in matched and "urgency" in matched:
            score += 1.0
            triggered_flags.append("combo_pressure")

        if "ai_vague" in matched and "no_structure" in matched:
            score += 1.0
            triggered_flags.append("combo_ai_fake")

        if re.search(r"(earn based on|per click|per engagement|referral)", text_lower):
            score += 1.0
            triggered_flags.append("affiliate_scam")

        if "external_contact" in matched:
            score += 0.8

        # Structural penalties
        if not re.search(r"(python|java|sql|excel|aws|react)", text_lower):
            score += 0.3
            triggered_flags.append("no_skills")

        if not re.search(r"(inc|ltd|corp|technologies|solutions)", text_lower):
            score += 0.2
            triggered_flags.append("weak_company")

        # Normalize to 0–1
        rule_score = min(score / 3.0, 1.0)

        return RuleEngineOutput(
            rule_score=rule_score,
            active_rules=active_rules,
            triggered_flags=triggered_flags,
        )