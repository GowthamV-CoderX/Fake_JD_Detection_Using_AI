"""
data/generate_dataset.py
Generates a realistic synthetic dataset of real vs fake job descriptions.
In production, replace/augment with:
  - Kaggle "Fake Job Postings" dataset (EMSCAD): 
    https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction
  - Indeed scrapes (real JDs)
  - Scam job forums / FTC complaint data (fake JDs)
"""

import json
import random
import pandas as pd
from pathlib import Path

random.seed(42)

# ──────────────────────────────────────────────
# REAL JD TEMPLATES
# ──────────────────────────────────────────────
REAL_JDS = [
    {
        "title": "Senior Software Engineer – Backend",
        "company": "Infosys Ltd",
        "location": "Pune, Maharashtra",
        "salary": "₹18–28 LPA",
        "text": (
            "Infosys is looking for a Senior Software Engineer to join our Cloud Platform team in Pune. "
            "You will design and build scalable microservices using Python and Go, collaborate with "
            "cross-functional teams, and participate in code reviews.\n\n"
            "Requirements:\n"
            "- 4+ years of backend development experience\n"
            "- Proficiency in Python or Go\n"
            "- Experience with AWS, Kubernetes, and Docker\n"
            "- Strong understanding of REST APIs and gRPC\n"
            "- B.Tech / M.Tech in Computer Science or related field\n\n"
            "Benefits: Medical insurance, 30 days PTO, ESOPs, flexible work-from-home policy.\n"
            "Apply at: careers.infosys.com | hr.pune@infosys.com"
        ),
        "label": 0,
    },
    {
        "title": "Data Analyst – Growth Analytics",
        "company": "Swiggy",
        "location": "Bengaluru, Karnataka",
        "salary": "₹10–16 LPA",
        "text": (
            "Swiggy's Growth Analytics team is hiring a Data Analyst to derive insights that drive "
            "user acquisition and retention strategies.\n\n"
            "What you'll do:\n"
            "- Build dashboards using Tableau and Metabase\n"
            "- Write complex SQL queries against our Redshift warehouse\n"
            "- Partner with product and marketing on A/B test design\n"
            "- Present findings to senior leadership weekly\n\n"
            "What we need:\n"
            "- 2+ years in analytics or data science\n"
            "- Expert SQL skills; Python (pandas) is a plus\n"
            "- Experience with BI tools\n"
            "- Strong communication skills\n\n"
            "CTC: ₹10–16 LPA + performance bonus\n"
            "Apply via LinkedIn or talent@swiggy.in"
        ),
        "label": 0,
    },
    {
        "title": "Product Manager – Fintech",
        "company": "Razorpay",
        "location": "Bengaluru (Hybrid)",
        "salary": "₹25–40 LPA",
        "text": (
            "Razorpay is seeking an experienced Product Manager to lead our Payment Gateway product line. "
            "You will own the roadmap, write PRDs, and work with engineering and design to ship features "
            "that serve 8M+ businesses.\n\n"
            "Responsibilities:\n"
            "- Define and prioritize product roadmap based on data and customer feedback\n"
            "- Coordinate with compliance, risk, and engineering teams\n"
            "- Drive 0-to-1 feature launches end-to-end\n\n"
            "Requirements:\n"
            "- 4–7 years of product management experience\n"
            "- Deep understanding of payments, UPI, or fintech ecosystem\n"
            "- MBA / B.Tech from a top-tier institute preferred\n\n"
            "Compensation: ₹25–40 LPA + ESOPs\n"
            "Contact: careers@razorpay.com | razorpay.com/careers"
        ),
        "label": 0,
    },
    {
        "title": "HR Executive – Talent Acquisition",
        "company": "Wipro Technologies",
        "location": "Hyderabad, Telangana",
        "salary": "₹4–6 LPA",
        "text": (
            "Wipro Technologies is hiring an HR Executive for Talent Acquisition to support end-to-end "
            "recruitment across IT and non-IT roles.\n\n"
            "Key Responsibilities:\n"
            "- Source candidates through job portals, LinkedIn, and referrals\n"
            "- Coordinate interviews with hiring managers\n"
            "- Manage offer letters and onboarding\n"
            "- Maintain MIS reports using Excel\n\n"
            "Eligibility:\n"
            "- MBA in HR or equivalent\n"
            "- 1–3 years of recruitment experience\n"
            "- Good communication skills in English and Telugu\n\n"
            "Salary: ₹4–6 LPA + PF + Health Insurance\n"
            "Email CV to: talent.hyd@wipro.com"
        ),
        "label": 0,
    },
    {
        "title": "DevOps Engineer",
        "company": "Freshworks Inc.",
        "location": "Chennai, Tamil Nadu",
        "salary": "₹12–22 LPA",
        "text": (
            "Freshworks is looking for a DevOps Engineer to strengthen our infrastructure reliability "
            "and CI/CD pipelines for SaaS products serving global customers.\n\n"
            "Responsibilities:\n"
            "- Manage and scale AWS infrastructure using Terraform\n"
            "- Build and maintain CI/CD pipelines (Jenkins, GitHub Actions)\n"
            "- Implement observability using Datadog and PagerDuty\n"
            "- Collaborate with development teams on release management\n\n"
            "Requirements:\n"
            "- 3+ years in DevOps/SRE roles\n"
            "- Strong AWS, Linux, Kubernetes skills\n"
            "- Scripting in Bash or Python\n\n"
            "Package: ₹12–22 LPA + Stock options\n"
            "Apply: jobs.freshworks.com"
        ),
        "label": 0,
    },
    {
        "title": "Content Writer – B2B SaaS",
        "company": "Zoho Corporation",
        "location": "Chennai / Remote",
        "salary": "₹5–9 LPA",
        "text": (
            "Zoho Corporation is looking for a Content Writer to create long-form blog posts, "
            "whitepapers, and case studies for our CRM and marketing automation products.\n\n"
            "Responsibilities:\n"
            "- Write 2–3 SEO-optimized articles per week\n"
            "- Research industry trends and translate them into reader-friendly content\n"
            "- Collaborate with product marketing on campaign assets\n\n"
            "Requirements:\n"
            "- 2+ years of B2B content writing experience\n"
            "- Familiarity with SEO tools (Ahrefs, SEMrush)\n"
            "- Excellent written English\n\n"
            "CTC: ₹5–9 LPA\n"
            "Apply at: zoho.com/careers or content-hiring@zoho.com"
        ),
        "label": 0,
    },
]

# ──────────────────────────────────────────────
# FAKE / SCAM JD TEMPLATES
# ──────────────────────────────────────────────
FAKE_JDS = [
    {
        "title": "Work From Home – Daily Payment",
        "company": "Online Earn Solutions",
        "location": "Anywhere, India",
        "salary": "₹50,000–1,00,000/week",
        "text": (
            "URGENT HIRING!! Work From Home Jobs Available NOW!! 🔥🔥\n\n"
            "Earn ₹50,000 to ₹1,00,000 per week sitting at home! No experience needed! "
            "Just 2 hours of work daily! 100% GUARANTEED payment every day!\n\n"
            "What you need to do:\n"
            "- Copy-paste simple tasks\n"
            "- Like and share social media posts\n"
            "- Data entry (typing speed doesn't matter)\n\n"
            "ANYONE CAN DO IT!! Students, housewives, retired persons welcome!\n\n"
            "HURRY!! Only 10 seats left!! Registration closes tonight!!\n\n"
            "Send your details to: earnonline2024@gmail.com or WhatsApp 9XXXXXXXXX\n"
            "Registration fee: Only ₹499 (refundable after first week!)"
        ),
        "label": 1,
    },
    {
        "title": "Digital Marketing Executive – Immediate Joining",
        "company": "Global Digital Services",
        "location": "All India",
        "salary": "₹80,000/month + unlimited commission",
        "text": (
            "We are a fast-growing digital marketing company looking for freshers and experienced candidates.\n\n"
            "Salary: ₹80,000/month GUARANTEED + unlimited commission!\n"
            "No target pressure. No office required.\n\n"
            "Profile:\n"
            "- Age 18–45\n"
            "- Basic smartphone knowledge\n"
            "- Willing to work hard\n\n"
            "Note: You will be required to purchase our starter kit worth ₹2,999 to begin your work. "
            "This is fully refundable after 30 days.\n\n"
            "APPLY FAST!! We are hiring today only!!\n"
            "Contact: globaldigital.hr@yahoo.com | Call/WhatsApp: 8XXXXXXXXX\n"
            "Note: Do not share this post publicly. Only selected candidates will be informed."
        ),
        "label": 1,
    },
    {
        "title": "Customer Support Executive",
        "company": "XYZ International BPO",
        "location": "Work From Home",
        "salary": "₹35,000/month",
        "text": (
            "Openings available for Customer Support!! 🌟\n\n"
            "Earn ₹35,000/month easily from home! No degree required! "
            "Company will provide training. International clients!\n\n"
            "Requirements:\n"
            "- Must have laptop/PC\n"
            "- Internet connection\n"
            "- Age 18+\n\n"
            "Process:\n"
            "1. Send CV\n"
            "2. Pay ₹1,500 security deposit (refunded with first salary)\n"
            "3. Get login credentials\n"
            "4. Start earning from Day 1!\n\n"
            "Limited seats! Contact NOW: xyzibpo_hr@gmail.com\n"
            "We are not responsible for any fraud. Beware of fake agents."
        ),
        "label": 1,
    },
    {
        "title": "Business Development Associate",
        "company": "Future Wealth Advisors",
        "location": "Pan India",
        "salary": "₹15,000–₹5,00,000/month",
        "text": (
            "Join our growing team and build a 6-figure income!! 💰💰\n\n"
            "We are looking for motivated individuals to join our network marketing team. "
            "There is NO upper limit to your earnings!\n\n"
            "What you'll do:\n"
            "- Recruit new members to our network\n"
            "- Promote our exclusive investment products\n"
            "- Attend weekly motivation sessions\n\n"
            "Investment required: ₹5,000–₹25,000 to join different levels\n"
            "Once you join, you earn from EVERY person you bring in!\n\n"
            "Top earners make ₹5 lakh/month! Be the next success story!\n\n"
            "WhatsApp NOW: 7XXXXXXXXX | futurewealthhrd@gmail.com\n"
            "This is a limited time opportunity. Don't let it slip away!"
        ),
        "label": 1,
    },
    {
        "title": "Part Time Job – Packaging Work",
        "company": "Home Industries",
        "location": "Mumbai / Remote",
        "salary": "₹20,000–₹40,000/month",
        "text": (
            "Home based packing work available! Earn from home without investment (almost)!\n\n"
            "Work: Pack spoons, candles, envelopes at home and courier back to us.\n"
            "Payment: Per piece basis. Earn ₹20k–40k monthly!\n\n"
            "Requirements:\n"
            "- Minimum 8 hours per day\n"
            "- Security deposit ₹2,000 (for raw materials)\n\n"
            "Note: Deposit is refundable after first consignment delivery.\n\n"
            "Contact: homeindustries.work@gmail.com | 9XXXXXXXXX\n"
            "Hurry! Only few vacancies left in your area!"
        ),
        "label": 1,
    },
    {
        "title": "Online Tutor / Academic Writer",
        "company": "EduHelp Solutions",
        "location": "Work From Home",
        "salary": "₹1,000–₹5,000/hour",
        "text": (
            "Earn ₹1000-5000 per hour doing academic writing!! Top opportunity for graduates!!\n\n"
            "We need people to:\n"
            "- Write assignments for students abroad\n"
            "- Complete online exams\n"
            "- Solve question papers\n\n"
            "Pay is per task. The more you do, the more you earn!\n\n"
            "You must:\n"
            "- Have any degree\n"
            "- Pay ₹999 registration fee (training materials included)\n\n"
            "URGENT REQUIREMENT!! 50 slots only!\n"
            "eduhelp.assignments@gmail.com | WhatsApp: 6XXXXXXXXX"
        ),
        "label": 1,
    },
]


def augment_jd(jd: dict, variation: int) -> dict:
    """Create slight textual variations to expand dataset size."""
    augmented = jd.copy()
    text = jd["text"]

    substitutions = [
        ("URGENT", "IMMEDIATE"),
        ("NOW", "TODAY"),
        ("HURRY", "ACT FAST"),
        ("guaranteed", "assured"),
        ("easy", "simple"),
        ("₹", "Rs."),
    ]

    if variation % 2 == 0 and substitutions:
        old, new = substitutions[variation % len(substitutions)]
        text = text.replace(old, new)

    augmented["text"] = text
    augmented["variation"] = variation
    return augmented


def generate_dataset(output_path: str = "data/raw/jd_dataset.csv"):
    records = []

    # Original samples
    for jd in REAL_JDS + FAKE_JDS:
        records.append({
            "title": jd["title"],
            "company": jd["company"],
            "location": jd["location"],
            "salary_range": jd["salary"],
            "description": jd["text"],
            "label": jd["label"],  # 0=real, 1=fake
        })

    # Augmented variations
    for i in range(1, 5):
        for jd in REAL_JDS:
            aug = augment_jd(jd, i)
            records.append({
                "title": aug["title"],
                "company": aug["company"],
                "location": aug["location"],
                "salary_range": aug["salary"],
                "description": aug["text"],
                "label": aug["label"],
            })
        for jd in FAKE_JDS:
            aug = augment_jd(jd, i)
            records.append({
                "title": aug["title"],
                "company": aug["company"],
                "location": aug["location"],
                "salary_range": aug["salary"],
                "description": aug["text"],
                "label": aug["label"],
            })

    df = pd.DataFrame(records)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset saved: {len(df)} records ({df['label'].sum()} fake, {(df['label']==0).sum()} real)")
    return df


if __name__ == "__main__":
    generate_dataset()
