"""
rule_engine.py  —  v3: Adversarial AI Fake Job Detection
Location: rule_engine.py (repo root)

KEY IMPROVEMENTS in v3:
  - NEW: _evaluate_adversarial_ai_fake() — detects AI-crafted fake jobs that
    look professional but contain subtle inconsistencies scammers use
  - NEW: coherence scoring — checks if role/salary/requirements actually match
  - NEW: vagueness scoring — real jobs are specific; AI fakes are fluently vague
  - NEW: structural deception patterns — things scammers add to look legit
  - IMPROVED: ai_score now feeds into FAKE verdict when combined with
    adversarial signals (not just "AI-GENERATED" neutral verdict)
  - 25 scam rules retained + 8 new adversarial checks
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple
import re
import math


@dataclass
class Rule:
    name: str
    severity: float
    pattern: str
    description: str


@dataclass
class RuleEngineOutput:
    rule_score: float
    ai_score: float
    adversarial_score: float                              # NEW
    adversarial_signals: List[str] = field(default_factory=list)  # NEW
    ai_signals: List[str] = field(default_factory=list)
    active_rules: List[Rule] = field(default_factory=list)
    triggered_flags: List[str] = field(default_factory=list)
    flag_descriptions: List[str] = field(default_factory=list)
    is_adversarial_fake: bool = False                    # NEW


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

        # ── AI-Generated Text Detection Phrases ───────────────────────────────
        self.ai_rules = [
            (r"\bwe are (looking for|seeking) a? ?(passionate|dynamic|motivated|talented|driven)\b", "LLM opener phrase"),
            (r"\bjoin our (dynamic|growing|innovative|passionate|talented) team\b", "LLM team phrase"),
            (r"\bfast-?paced (environment|setting|workplace)\b", "LLM filler: fast-paced"),
            (r"\bexcellent (communication|interpersonal|organizational) skills\b", "LLM filler: soft skills"),
            (r"\bstrong attention to detail\b", "LLM filler: attention to detail"),
            (r"\bproven track record\b", "LLM filler: track record"),
            (r"\bresults?-driven\b", "LLM filler: results-driven"),
            (r"\bdata-driven\b", "LLM filler: data-driven"),
            (r"\bcross-functional (teams?|collaboration|projects?)\b", "LLM filler: cross-functional"),
            (r"\bseamlessly\b", "LLM filler: seamlessly"),
            (r"\brobust (experience|background|skill set|knowledge)\b", "LLM filler: robust"),
            (r"\bkey responsibilities (include|are)\b", "LLM section header"),
            (r"\bresponsibilities include but are not limited to\b", "LLM boilerplate"),
            (r"\bwe offer a competitive (salary|compensation|package)\b", "LLM offer phrase"),
            (r"\bequal opportunity employer\b", "LLM boilerplate: EEO"),
            (r"\bwork[- ]life balance\b", "LLM filler: work-life balance"),
            (r"\bgrowth mindset\b", "LLM filler: growth mindset"),
            (r"\bstakeholders?\b", "LLM filler: stakeholder"),
            (r"\bend-to-end\b", "LLM filler: end-to-end"),
            (r"\bbest-in-class\b", "LLM filler: best-in-class"),
            (r"\bworld-class\b", "LLM filler: world-class"),
            (r"\b(exciting|unique) opportunity\b", "LLM filler: exciting opportunity"),
            (r"\brapidly (growing|expanding|scaling)\b", "LLM filler: rapidly growing"),
            (r"\bsolution-?oriented\b", "LLM filler: solution-oriented"),
            (r"\bcritical thinking\b", "LLM filler: critical thinking"),
            (r"\bproblem[- ]solving skills\b", "LLM filler: problem-solving"),
            (r"\btime management skills\b", "LLM filler: time management"),
            (r"\bself[- ]starter\b", "LLM filler: self-starter"),
            (r"\bteam player\b", "LLM filler: team player"),
            (r"\bgo[- ]getter\b", "LLM filler: go-getter"),
            (r"\bpassionate (about|individual|professional)\b", "LLM filler: passionate"),
            (r"\bsynergize?\b", "LLM filler: synergy"),
            (r"\bleverage (our|the|your|this)\b", "LLM filler: leverage"),
            (r"\bholistic approach\b", "LLM filler: holistic"),
            (r"\bscalable solution\b", "LLM filler: scalable"),
            (r"\bthought leader\b", "LLM filler: thought leader"),
            (r"\bmove the needle\b", "LLM filler: move the needle"),
            (r"\bbring to the table\b", "LLM filler: bring to the table"),
            (r"(about the role|about the position|about this role)\s*[\n\r:]", "LLM section: About the Role"),
            (r"(what you.ll do|what you will do)\s*[\n\r:]", "LLM section: What You'll Do"),
            (r"(what you.ll bring|what you bring|what you will bring)\s*[\n\r:]", "LLM section: What You'll Bring"),
            (r"(minimum qualifications|preferred qualifications)\s*[\n\r:]", "LLM section: Qualifications"),
            (r"(nice to have|bonus points)\s*[\n\r:]", "LLM section: Nice to Have"),
            (r"(what we offer|why join us|perks (and|&) benefits)\s*[\n\r:]", "LLM section: What We Offer"),
        ]

        # ── Adversarial AI Fake Patterns ──────────────────────────────────────
        # These are signals that AI uses to make fake jobs LOOK real.
        # They're not individually damning — but combined with AI score, they are.
        self.adversarial_patterns = [

            # 1. Fake specificity — AI adds numbers/percentages to sound real but
            #    they're always round, impressive, and unverifiable
            (r"\b(over|more than|up to|around)\s+\d{2,3}[,.]?\d*\s*"
             r"(employees?|team members?|clients?|countries|offices|years?)\b",
             "Fake company scale claim (AI adds impressive-sounding numbers)", 0.12),

            (r"\b\d+%\s+(growth|increase|improvement|efficiency|retention|satisfaction)\b",
             "Suspiciously precise metric (AI-generated credibility signal)", 0.14),

            (r"\b(fortune\s+500|top\s+\d+|global\s+leader|industry.leading|market.leading"
             r"|award.winning|iso\s+certified|nasdaq|sensex.listed)\b",
             "Unverifiable prestige claim (common AI legitimacy tactic)", 0.15),

            # 2. Vague-but-sophisticated role descriptions — AI writes fluently but
            #    avoids specifics that would require real domain knowledge
            (r"\b(drive(s|d)?\s+(growth|impact|innovation|results?|strategy|initiatives?))\b",
             "Generic 'drive' verb — AI avoids specific action words", 0.10),

            (r"\b(partner(ing)?\s+with\s+(cross.functional|internal|key|various)\s+teams?)\b",
             "Vague collaboration phrasing (AI filler)", 0.10),

            (r"\b(ensure(s|d)?\s+(smooth|seamless|efficient|effective)\s+(operations?|processes?|delivery|execution))\b",
             "Generic process responsibility — AI lacks domain specifics", 0.10),

            (r"\b(contribute\s+to\s+(the\s+)?(overall|company.?wide|organizational)\s+(success|goals?|vision|strategy))\b",
             "Vague contribution framing — AI avoids concrete deliverables", 0.12),

            # 3. Compensation vagueness + prestige combo — AI offers "competitive"
            #    salary without real ranges (scammers avoid commitment)
            (r"\bcompetitive\s+(salary|compensation|package|pay|ctc)\b(?!.*(\d[\d,]*|\blpa\b|\bper\s+(month|annum|year)\b))",
             "Competitive salary with no actual range — typical AI evasion", 0.18),

            (r"\b(salary\s+(commensurate|based)\s+(with|on)\s+experience)\b",
             "Salary-dodging phrase — avoids commitment to real figures", 0.14),

            # 4. Contact/apply evasion — AI-crafted fakes often have
            #    professional-looking but commitment-free application steps
            (r"\b(reach\s+out|connect\s+with\s+us|drop\s+(your\s+)?resume|send\s+(your\s+)?cv)\s+"
             r"(at|to|via|on)\s+\S+@(gmail|yahoo|hotmail|outlook)\b",
             "Professional tone but personal email — AI scam pattern", 0.25),

            (r"\bapply\s+(now|today|here|directly)\s*[:\-]?\s*(https?://\S+|bit\.ly|tinyurl)\b",
             "Apply via unverifiable short URL — AI fake legitimacy trick", 0.20),

            # 5. Responsibility-requirement mismatch — AI generates both sections
            #    but they often don't coherently connect (senior role + junior reqs)
            (r"(senior|lead|head\s+of|principal|director|vp\s+of).*?"
             r"(0[-–]2\s+years?|fresher|fresh\s+graduate|recent\s+graduate|entry.level)",
             "Senior title with entry-level requirements — incoherent (AI mismatch)", 0.30),

            (r"(intern|trainee|associate|junior|entry.level).*?"
             r"(manage\s+(team|budget|p&l|department)|lead\s+(team|cross.functional)|strategic\s+decisions?)",
             "Junior role with senior responsibilities — incoherent (AI mismatch)", 0.28),

            # 6. Location incoherence — AI sometimes combines WFH with location
            #    specificity in ways that don't make sense
            (r"\b(fully\s+remote|100%\s+remote|work\s+from\s+anywhere)\b.*?"
             r"\b(must\s+(be\s+)?(based|located|reside|relocate)\s+(in|to|at))\b",
             "Remote role requiring physical location — AI contradiction", 0.22),

            # 7. Benefits inflation — AI always lists the same impressive benefits
            #    that real startups/small companies can't actually offer
            (r"(health\s+(insurance|coverage)|dental|vision|401k|esop|stock\s+options?|equity)\b"
             r".*?(flexible|unlimited)\s+(pto|leaves?|vacation|time\s+off)",
             "Full benefits + unlimited PTO — AI benefit inflation pattern", 0.12),

            # 8. Fake urgency in professional language — scammers learned that
            #    "urgent" is flagged, so AI uses sophisticated urgency instead
            (r"\b(immediate(ly)?|asap)\s+(start|joining|onboarding|availability\s+preferred)\b",
             "Sophisticated urgency phrasing (AI-polished pressure tactic)", 0.16),

            (r"\bwe\s+(are\s+)?looking\s+to\s+(fill\s+(this|the)\s+position|hire)\s+"
             r"(immediately|urgently|as\s+soon\s+as\s+possible|within\s+(\d+\s+)?(days?|week))\b",
             "Urgency framing in professional language — AI scam polish", 0.16),

            # 9. Zero-contact company info — AI gives company name but no real
            #    verifiable details (website, LinkedIn, registration number)
            (r"^(?!.*(\.com|\.in|\.io|\.co\.in|linkedin\.com|glassdoor|indeed"
             r"|crunchbase|cin\s*:|gstin\s*:|registration\s*no))",
             "No verifiable company URL or registration — AI fake legitimacy gap", 0.15),

            # 10. Over-emphasis on culture without substance — AI defaults to
            #     culture talk when it has no real company info
            (r"\b(our\s+)?(culture|values?|mission|vision)\s+(is|are|focuses?|centers?|revolves?)\b"
             r".*?\b(innovation|collaboration|excellence|integrity|diversity|inclusion)\b",
             "Culture/values section with only buzzwords — AI padding", 0.10),

            # 11. Suspiciously polished equal opportunity statement on an
            #     otherwise thin post — AI always adds EEO to seem legitimate
            (r"\bequal\s+(opportunity|employment)\b.*?"
             r"\b(race|color|religion|sex|gender|national\s+origin|disability|veteran)\b",
             "Boilerplate EEO statement on suspicious post — AI legitimacy tactic", 0.12),

            # 12. Commission-heavy compensation disguised professionally
            (r"\b(ote|on.target\s+earnings?|variable\s+(pay|component|compensation))\b"
             r".*?\b(\d{2,3}%|majority|bulk|most\s+of)\b",
             "Mostly variable/commission pay disguised in professional language", 0.18),
        ]

    # ── Scam Rule Evaluator ───────────────────────────────────────────────────
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

        rule_score = min(score / 4.0, 1.0)

        # ── AI-Generated Text Scoring ─────────────────────────────────────────
        ai_score, ai_signals = self._evaluate_ai_text(text)

        # ── Adversarial AI Fake Scoring ───────────────────────────────────────
        adversarial_score, adversarial_signals = self._evaluate_adversarial_ai_fake(text, ai_score)

        # Merge adversarial flags into triggered_flags for display
        if adversarial_signals:
            for sig in adversarial_signals:
                if sig not in flag_descriptions:
                    flag_descriptions.append(sig)

        return RuleEngineOutput(
            rule_score=rule_score,
            ai_score=ai_score,
            adversarial_score=adversarial_score,
            adversarial_signals=adversarial_signals,
            ai_signals=ai_signals,
            active_rules=active_rules,
            triggered_flags=triggered_flags,
            flag_descriptions=flag_descriptions,
            is_adversarial_fake=adversarial_score >= 0.45,
        )

    # ── AI-Generated Text Evaluator ───────────────────────────────────────────
    def _evaluate_ai_text(self, text: str) -> Tuple[float, List[str]]:
        text_lower = text.lower()
        signals: List[str] = []
        points = 0.0

        phrase_hits = 0
        for pattern, description in self.ai_rules:
            weight = 2 if "section" in description.lower() else 1
            if re.search(pattern, text_lower, re.DOTALL):
                phrase_hits += weight
                signals.append(description)

        if phrase_hits >= 12:
            points += 0.40
        elif phrase_hits >= 7:
            points += 0.28
        elif phrase_hits >= 4:
            points += 0.16
        elif phrase_hits >= 2:
            points += 0.06

        # Sentence length uniformity
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 10]
        if len(sentences) >= 5:
            lengths = [len(s.split()) for s in sentences]
            mean = sum(lengths) / len(lengths)
            variance = sum((l - mean) ** 2 for l in lengths) / len(lengths)
            if variance < 12:
                points += 0.18
                signals.append("Very uniform sentence lengths (AI-like writing)")
            elif variance < 25:
                points += 0.08
                signals.append("Moderately uniform sentence lengths")

        # Bullet density
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if lines:
            bullet_lines = sum(1 for l in lines if re.match(r'^[\-\*\•\✓\→]|^\d+[\.\)]', l))
            bullet_ratio = bullet_lines / len(lines)
            if bullet_ratio > 0.45:
                points += 0.14
                signals.append(f"High bullet density ({bullet_ratio:.0%}) — AI-structured layout")
            elif bullet_ratio > 0.25:
                points += 0.06

        # Grammar consistency
        if len(sentences) >= 4:
            well_formed = sum(1 for s in sentences if s and s[0].isupper() and s[-1] in '.!?')
            if (well_formed / len(sentences)) > 0.88:
                points += 0.10
                signals.append("Unusually consistent grammar structure (AI-like)")

        # Vocabulary diversity
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        if len(words) > 60:
            ttr = len(set(words)) / len(words)
            if 0.38 <= ttr <= 0.65:
                points += 0.06
                signals.append(f"AI-typical vocabulary diversity (TTR={ttr:.2f})")

        # Human scam signals cancel AI score
        scam_cancels = [
            r"\b(whatsapp|telegram|registration\s+fee|pay\s+to\s+join)\b",
            r"\b(earn\s+₹[\d,]+\s*per\s*(day|week)|guaranteed\s+income)\b",
            r"(!{3,}|([A-Z]{4,}\s+){3,})",
            r"\b(ghar\s+baithe|paisa\s+kamao|daily\s+earning)\b",
        ]
        cancel_hits = sum(1 for p in scam_cancels if re.search(p, text_lower))
        if cancel_hits >= 2:
            points -= 0.30
            signals.append("Strong human scam signals — not AI-generated")
        elif cancel_hits == 1:
            points -= 0.12

        ai_score = round(max(0.0, min(1.0, points)), 3)
        return ai_score, signals

    # ── Adversarial AI Fake Detector ─────────────────────────────────────────
    def _evaluate_adversarial_ai_fake(self, text: str, ai_score: float) -> Tuple[float, List[str]]:
        """
        Detects AI-crafted fake jobs that look professional.

        These posts:
        - Pass all normal scam rules (no obvious red flags)
        - Are written in polished, professional English
        - But contain subtle inconsistencies + evasions that real jobs don't have

        Strategy:
        1. Check adversarial pattern hits
        2. Check internal coherence (role/salary/requirements match)
        3. Check specificity ratio (real jobs have real domain specifics)
        4. Combine with AI score — high AI + adversarial = FAKE not just AI-GENERATED

        Returns (adversarial_score 0.0-1.0, signal descriptions)
        """
        text_lower = text.lower()
        signals: List[str] = []
        points = 0.0

        # Only run adversarial check if AI score is elevated
        # Low AI score = human-written (genuine or obvious scam) — skip this check
        if ai_score < 0.15:
            return 0.0, []

        # 1. Pattern matching
        pattern_points = 0.0
        for pattern, description, weight in self.adversarial_patterns:
            if re.search(pattern, text, re.DOTALL | re.IGNORECASE):
                pattern_points += weight
                signals.append(f"⚠ {description}")

        points += min(pattern_points, 0.55)

        # 2. Specificity check — real jobs name specific tools, tech, or processes
        #    AI fakes describe tasks in generic action verbs without domain substance
        specificity_score = self._check_specificity(text_lower)
        if specificity_score < 0.15:
            points += 0.20
            signals.append("Critically low specificity — job tasks described in generic verbs only")
        elif specificity_score < 0.28:
            points += 0.10
            signals.append("Low specificity — lacks concrete tools/technologies/metrics")

        # 3. Coherence check — do the role title, responsibilities and requirements align?
        coherence_penalty = self._check_coherence(text_lower)
        if coherence_penalty > 0:
            points += coherence_penalty
            signals.append("Role-requirements mismatch — AI-generated content inconsistency")

        # 4. Vagueness ratio — ratio of vague action verbs to specific domain terms
        vagueness = self._check_vagueness_ratio(text_lower)
        if vagueness > 0.70:
            points += 0.18
            signals.append(f"High vagueness ratio ({vagueness:.0%}) — AI uses generic verbs to fake expertise")
        elif vagueness > 0.50:
            points += 0.08

        # 5. Missing verifiable anchors — real job posts have at least some
        #    verifiable info; AI fakes often have none
        anchor_score = self._check_verifiable_anchors(text)
        if anchor_score == 0:
            points += 0.18
            signals.append("Zero verifiable anchors (no company URL, CIN, LinkedIn, location, team size)")
        elif anchor_score == 1:
            points += 0.08

        # 6. Salary-role coherence
        salary_penalty = self._check_salary_role_coherence(text_lower)
        if salary_penalty > 0:
            points += salary_penalty
            signals.append("Salary range inconsistent with role seniority — AI compensation mismatch")

        # 7. AI score amplifier — the higher the AI score, the more
        #    adversarial signals matter (adversarial only matters if AI-written)
        ai_amplifier = 1.0 + (ai_score - 0.15) * 1.2  # scales 1.0 to ~2.0
        points = points * ai_amplifier

        adversarial_score = round(max(0.0, min(1.0, points)), 3)
        return adversarial_score, signals

    def _check_specificity(self, text: str) -> float:
        """
        Ratio of domain-specific terms to total content words.
        Real jobs mention specific tools, technologies, processes.
        AI fakes describe roles in generic management/coordination verbs.
        """
        specific_terms = [
            # Tech
            r"\b(python|java|javascript|typescript|react|angular|vue|node\.?js|django|flask"
            r"|spring|kubernetes|docker|aws|gcp|azure|terraform|postgres|mysql|mongodb"
            r"|redis|kafka|spark|hadoop|tableau|power\s*bi|looker|dbt|airflow|mlflow"
            r"|tensorflow|pytorch|scikit.learn|pandas|numpy)\b",
            # Finance/Accounting
            r"\b(tally|quickbooks|sap|oracle\s+financials|ifrs|gaap|gst|tds|p&l|ebitda"
            r"|balance\s+sheet|accounts\s+receivable|accounts\s+payable|cagr)\b",
            # Marketing
            r"\b(google\s+ads|facebook\s+ads|meta\s+ads|seo|sem|cpc|ctr|roas|hubspot"
            r"|salesforce|marketo|mailchimp|ahrefs|semrush|google\s+analytics|ga4)\b",
            # Healthcare
            r"\b(emr|ehr|hipaa|icd.10|cpt\s+codes|phlebotomy|radiology|pathology"
            r"|pharmacovigilance|clinical\s+trials?|gcp\s+guidelines)\b",
            # Legal
            r"\b(ipc|crpc|civil\s+procedure|arbitration|due\s+diligence|m&a|sebi"
            r"|companies\s+act|patent\s+filing|trademark|nda|shareholders\s+agreement)\b",
            # HR
            r"\b(hris|ats|workday|successfactors|darwinbox|keka|pf\s+compliance"
            r"|gratuity|esic|form\s+16|ctc\s+structure|joining\s+formalities)\b",
            # Design
            r"\b(figma|sketch|adobe\s+(xd|illustrator|photoshop|premiere|after\s+effects)"
            r"|invision|zeplin|ui/ux|wireframe|prototype|design\s+system|accessibility)\b",
        ]
        word_count = len(re.findall(r'\b[a-zA-Z]{3,}\b', text))
        if word_count == 0:
            return 0.0
        specific_count = sum(
            len(re.findall(p, text, re.IGNORECASE))
            for p in specific_terms
        )
        return min(specific_count / max(word_count * 0.1, 1), 1.0)

    def _check_coherence(self, text: str) -> float:
        """
        Returns a penalty (0.0-0.35) if role title and requirements are incoherent.
        """
        penalty = 0.0
        senior_titles = r"\b(senior|lead|head|principal|director|vp|vice\s+president|manager|architect)\b"
        junior_reqs = r"\b(0[-–]\s*[12]\s+year|fresher|fresh\s+graduate|no\s+experience|entry.level|trainee)\b"
        junior_titles = r"\b(intern|trainee|junior|associate|entry.level|fresher)\b"
        senior_resp = r"\b(manage\s+team|p&l\s+responsibility|budget\s+ownership|strategic\s+planning|board\s+level)\b"

        if re.search(senior_titles, text) and re.search(junior_reqs, text):
            penalty += 0.30
        if re.search(junior_titles, text) and re.search(senior_resp, text):
            penalty += 0.28
        return penalty

    def _check_vagueness_ratio(self, text: str) -> float:
        """
        Ratio of generic action verbs to total verbs.
        AI fakes use high ratio of vague verbs; real jobs mix in domain-specific verbs.
        """
        vague_verbs = [
            r"\b(drive|ensure|support|manage|lead|oversee|coordinate|facilitate"
            r"|collaborate|engage|leverage|optimize|streamline|enhance|deliver"
            r"|execute|implement|develop|build|grow|scale|transform|align)\b"
        ]
        specific_verbs = [
            r"\b(deploy|debug|refactor|migrate|configure|architect|query|index"
            r"|audit|reconcile|file|litigate|prescribe|diagnose|underwrite"
            r"|backtest|model|forecast|render|animate|typeset|compile|serialize)\b"
        ]
        vague_count = sum(len(re.findall(p, text)) for p in vague_verbs)
        specific_count = sum(len(re.findall(p, text)) for p in specific_verbs)
        total = vague_count + specific_count
        if total == 0:
            return 0.5  # neutral
        return vague_count / total

    def _check_verifiable_anchors(self, text: str) -> int:
        """
        Count verifiable anchors. Real companies have at least 1-2.
        Returns count (0 = worst, 3+ = real).
        """
        anchors = [
            r"https?://\S+\.(com|in|io|co\.in|org|net)",  # company website
            r"\b(linkedin\.com|glassdoor\.com|ambitionbox|indeed\.com)\b",
            r"\b(cin|gstin|registration\s+no\.?)\s*[:\-]?\s*[A-Z0-9]{10,}",
            r"\b(bengaluru|mumbai|hyderabad|chennai|pune|delhi|noida|gurgaon"
            r"|gurugram|kolkata|ahmedabad)\s*[\-,]\s*(karnataka|maharashtra|telangana"
            r"|tamil\s+nadu|delhi\s+ncr)",  # specific Indian city+state
            r"\b\d{3,4}\s+employees?\b",  # specific headcount
            r"\bfounded\s+in\s+(19|20)\d{2}\b",  # founding year
        ]
        return sum(1 for p in anchors if re.search(p, text, re.IGNORECASE))

    def _check_salary_role_coherence(self, text: str) -> float:
        """
        Returns penalty if salary range doesn't match expected role level.
        AI sometimes puts senior salaries on junior roles or vice versa to attract clicks.
        """
        # Extract any salary mention
        salary_match = re.search(
            r"(\d+(?:\.\d+)?)\s*(?:[-–to]+\s*(\d+(?:\.\d+)?))?\s*(lpa|lakh|lakhs?\s+per\s+annum|l\s+p\.?\s*a)",
            text, re.IGNORECASE
        )
        if not salary_match:
            return 0.0

        try:
            low = float(salary_match.group(1))
            high = float(salary_match.group(2)) if salary_match.group(2) else low
            avg_salary = (low + high) / 2
        except (ValueError, TypeError):
            return 0.0

        is_senior = bool(re.search(
            r"\b(senior|lead|head|director|principal|architect|manager|vp)\b", text, re.IGNORECASE
        ))
        is_junior = bool(re.search(
            r"\b(intern|fresher|trainee|entry.level|junior|associate)\b", text, re.IGNORECASE
        ))

        # Senior role with suspiciously low salary (bait-and-switch)
        if is_senior and avg_salary < 4:
            return 0.20
        # Junior role with suspiciously high salary (too good to be true)
        if is_junior and avg_salary > 25:
            return 0.22
        return 0.0