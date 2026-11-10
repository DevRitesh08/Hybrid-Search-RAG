"""Generate synthetic 'NovaStar Bank' docs for refusal testing."""

from __future__ import annotations

import json

from hybrid_rag.config import settings

DEMO_BANK_DIR = settings.data_raw_dir / "demo_bank"
REFUSAL_QUESTIONS_PATH = settings.golden_path.parent / "refusal_questions.jsonl"

DEMO_DOCS = {
    "accounts.md": """# NovaStar Bank: Account Types

NovaStar Bank offers two primary checking options:
1. **NovaStar Essential Checking**: Zero monthly maintenance fees with direct deposit ($500 minimum monthly), standard debit card, and free online bill pay.
2. **NovaStar Premier Checking**: $15 monthly fee waived with $10,000 combined balance, earns 0.75% APY, includes complimentary out-of-network ATM fee rebates up to $20/month.

Savings accounts include our High-Yield Savings Account earning 4.25% APY with no minimum balance requirement.
""",
    "fees.md": """# NovaStar Bank: Schedule of Fees

- **Overdraft Fee**: $29 per transaction (maximum 2 per day, waived for transactions under $5).
- **Wire Transfer (Domestic Outgoing)**: $25.
- **Wire Transfer (International Outgoing)**: $45.
- **Paper Statement Fee**: $3 per month (electronic statements are free).
- **Card Replacement (Standard)**: Free (3-5 business days).
- **Card Replacement (Rush Overnight)**: $30.
""",
    "security.md": """# NovaStar Bank: Security and Fraud Prevention

NovaStar Bank utilizes multi-factor authentication (MFA) via SMS, Authenticator App, or physical security key (WebAuthn).
Accounts are monitored 24/7 with automated fraud detection algorithms.
Zero Liability Protection guarantees you are not held responsible for unauthorized transactions reported within 60 days.
To report lost or stolen cards, contact the Fraud Operations Center immediately at 1-800-555-NOVA.
""",
    "loans.md": """# NovaStar Bank: Personal and Auto Loans

NovaStar provides fixed-rate personal loans ranging from $2,500 to $40,000 with repayment terms from 24 to 60 months.
APR rates vary from 6.99% to 19.99% based on credit score, debt-to-income ratio, and term length.
Auto loans for new vehicles start at 4.49% APR for up to 72 months.
Pre-qualification does not impact your credit score.
""",
    "support.md": """# NovaStar Bank: Customer Support and Hours

Customer Service hotline: 1-800-555-NOVA (6682).
Support hours:
- Monday through Friday: 7:00 AM – 9:00 PM Eastern Time.
- Saturday: 8:00 AM – 5:00 PM Eastern Time.
- Sunday: Closed (Automated Voice Banking available 24/7).

Secure messaging is accessible through the NovaStar Mobile App or online portal.
""",
}

UNANSWERABLE_QUESTIONS = [
    {
        "id": "ref_001",
        "question": "What is NovaStar Bank's interest rate for commercial construction loans over $50 million?",
        "expected": "refusal",
        "reason": "Commercial construction loans are not offered or documented by NovaStar Bank.",
    },
    {
        "id": "ref_002",
        "question": "How do I trade cryptocurrency options on NovaStar Bank's mobile app?",
        "expected": "refusal",
        "reason": "NovaStar Bank does not support cryptocurrency trading.",
    },
    {
        "id": "ref_003",
        "question": "What is the penalty for withdrawing from a NovaStar 10-year Certificate of Deposit?",
        "expected": "refusal",
        "reason": "Certificates of deposit are not documented in the provided corpus.",
    },
    {
        "id": "ref_004",
        "question": "Where is NovaStar Bank's branch located in Tokyo, Japan?",
        "expected": "refusal",
        "reason": "NovaStar Bank does not operate international branches.",
    },
    {
        "id": "ref_005",
        "question": "Can I pay for my NovaStar mortgage using gold bullion bars?",
        "expected": "refusal",
        "reason": "Fanciful out-of-domain unanswerable question.",
    },
]


def main() -> None:
    DEMO_BANK_DIR.mkdir(parents=True, exist_ok=True)
    for fname, content in DEMO_DOCS.items():
        (DEMO_BANK_DIR / fname).write_text(content.strip(), encoding="utf-8")
    print(f"✓ Created {len(DEMO_DOCS)} Demo Bank documents in {DEMO_BANK_DIR}")

    REFUSAL_QUESTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REFUSAL_QUESTIONS_PATH, "w", encoding="utf-8") as f:
        for q in UNANSWERABLE_QUESTIONS:
            f.write(json.dumps(q) + "\n")
    print(f"✓ Created refusal questions in {REFUSAL_QUESTIONS_PATH}")


if __name__ == "__main__":
    main()
