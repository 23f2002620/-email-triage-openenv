"""
Deterministic email fixtures for each task.
Using fixed seeds so graders are reproducible.
"""

from __future__ import annotations
from typing import List, Dict
from models import Email, EmailCategory


# ---------------------------------------------------------------------------
# TASK 1: Email Classification — single email per episode
# We expose 10 labelled emails so episodes can cycle through them.
# ---------------------------------------------------------------------------

CLASSIFY_EMAILS: List[Dict] = [
    {
        "id": "e001",
        "subject": "URGENT: Production server is down!",
        "body": (
            "Our production database is throwing 500 errors since 2 AM. "
            "Revenue is halting. We need immediate assistance from your DevOps team. "
            "This is a Sev-1 incident."
        ),
        "sender": "cto@bigclient.com",
        "timestamp": "2024-03-15T02:14:00Z",
        "label": EmailCategory.URGENT,
    },
    {
        "id": "e002",
        "subject": "Win a FREE iPhone — Click here NOW!!!",
        "body": (
            "Congratulations! You've been selected. Click the link to claim your "
            "prize. Limited time offer. Unsubscribe at spamlink.ru"
        ),
        "sender": "noreply@prizewinner9.net",
        "timestamp": "2024-03-15T08:00:00Z",
        "label": EmailCategory.SPAM,
    },
    {
        "id": "e003",
        "subject": "Question about invoice #4521",
        "body": (
            "Hi, I received my invoice but I think there might be a small error "
            "in the line items. Could you double-check when you get a chance? "
            "No rush, just want to make sure everything is correct."
        ),
        "sender": "accounting@smallcorp.com",
        "timestamp": "2024-03-15T09:30:00Z",
        "label": EmailCategory.NORMAL,
    },
    {
        "id": "e004",
        "subject": "Critical security breach detected — immediate action required",
        "body": (
            "Our monitoring system detected unauthorized access to your API keys "
            "at 11:45 PM. Keys may be compromised. Please rotate all credentials "
            "immediately and review the audit log. Contact security@ourteam.com."
        ),
        "sender": "security-alerts@ourplatform.com",
        "timestamp": "2024-03-15T23:50:00Z",
        "label": EmailCategory.URGENT,
    },
    {
        "id": "e005",
        "subject": "Monthly newsletter — March edition",
        "body": (
            "Welcome to our March newsletter! This month we cover product updates, "
            "upcoming webinars, and community highlights. Enjoy reading!"
        ),
        "sender": "newsletter@saascompany.com",
        "timestamp": "2024-03-15T10:00:00Z",
        "label": EmailCategory.SPAM,
    },
    {
        "id": "e006",
        "subject": "Meeting reschedule request",
        "body": (
            "Hi, something came up and I need to reschedule our Thursday 3pm meeting. "
            "Would Friday at 2pm work for you? Let me know."
        ),
        "sender": "partner@clientfirm.com",
        "timestamp": "2024-03-15T11:15:00Z",
        "label": EmailCategory.NORMAL,
    },
    {
        "id": "e007",
        "subject": "System outage — customers cannot log in",
        "body": (
            "Multiple enterprise customers are reporting they cannot authenticate. "
            "Our SSO integration appears broken after last night's deployment. "
            "Ticket #9821 opened. Escalating to P0."
        ),
        "sender": "oncall@ourinfra.com",
        "timestamp": "2024-03-15T07:02:00Z",
        "label": EmailCategory.URGENT,
    },
    {
        "id": "e008",
        "subject": "Request for feature documentation",
        "body": (
            "Hey team, I was wondering if there's any documentation on the new "
            "bulk export feature? I've been trying to find it in the knowledge base "
            "but couldn't locate it. Thanks in advance!"
        ),
        "sender": "user123@customer.io",
        "timestamp": "2024-03-15T14:20:00Z",
        "label": EmailCategory.NORMAL,
    },
    {
        "id": "e009",
        "subject": "You have been pre-approved for a $50,000 loan",
        "body": (
            "Dear valued customer, you qualify for our exclusive loan offer. "
            "No credit check needed. Apply in 5 minutes. Click: loan-scam.biz/apply"
        ),
        "sender": "offers@fastloan-deals.com",
        "timestamp": "2024-03-15T15:00:00Z",
        "label": EmailCategory.SPAM,
    },
    {
        "id": "e010",
        "subject": "Contract renewal — action needed by Friday",
        "body": (
            "Hi, our annual contract expires this Friday. I'd like to discuss renewal "
            "terms. Could someone from your account team reach out today? "
            "We're also considering other vendors so timing is important."
        ),
        "sender": "vp-ops@enterprise-client.com",
        "timestamp": "2024-03-15T09:00:00Z",
        "label": EmailCategory.NORMAL,
    },
]


# ---------------------------------------------------------------------------
# TASK 2: Email Prioritization — fixed inbox of 5 emails
# Correct priority order: urgent emails first, then normal, then spam.
# ---------------------------------------------------------------------------

PRIORITIZE_INBOX: List[Dict] = [
    {
        "id": "p001",
        "subject": "URGENT: Payment gateway down — transactions failing",
        "body": "All credit card transactions have been failing for the past 30 mins. Immediate fix needed.",
        "sender": "alerts@payments.io",
        "timestamp": "2024-03-15T14:00:00Z",
        "label": EmailCategory.URGENT,
        "priority_rank": 1,
    },
    {
        "id": "p002",
        "subject": "Can you send me the Q1 report?",
        "body": "Hi, whenever you get a chance could you forward the Q1 financial summary? Thanks.",
        "sender": "manager@corp.com",
        "timestamp": "2024-03-15T11:00:00Z",
        "label": EmailCategory.NORMAL,
        "priority_rank": 3,
    },
    {
        "id": "p003",
        "subject": "Critical: Data loss incident — backups missing",
        "body": "Nightly backup job failed silently for 3 days. Customer data at risk. Escalate now.",
        "sender": "dba@ourteam.com",
        "timestamp": "2024-03-15T06:00:00Z",
        "label": EmailCategory.URGENT,
        "priority_rank": 2,
    },
    {
        "id": "p004",
        "subject": "Exclusive deals just for you — 80% off everything",
        "body": "Don't miss out! Shop now at discountpalace.net. Offer expires midnight.",
        "sender": "deals@discountpalace.net",
        "timestamp": "2024-03-15T08:30:00Z",
        "label": EmailCategory.SPAM,
        "priority_rank": 5,
    },
    {
        "id": "p005",
        "subject": "Feedback on last week's workshop",
        "body": "Hey, just wanted to say the workshop was great. A few notes attached for your consideration.",
        "sender": "attendee@partner.org",
        "timestamp": "2024-03-15T12:30:00Z",
        "label": EmailCategory.NORMAL,
        "priority_rank": 4,
    },
]

# Correct priority order (highest to lowest)
CORRECT_PRIORITY_ORDER = ["p001", "p003", "p002", "p005", "p004"]


# ---------------------------------------------------------------------------
# TASK 3: Email Response Drafting
#
# Three different complaint emails so the task is replayable.
# RESPONSE_EMAILS is the full pool; RESPONSE_EMAIL is a convenience alias
# for index 0 (backwards-compatible with existing code that imports it).
# environment.py selects by email_index % len(RESPONSE_EMAILS).
# ---------------------------------------------------------------------------

RESPONSE_EMAILS: List[Dict] = [
    {
        "id": "r001",
        "subject": "Billing error — charged twice for subscription",
        "body": (
            "Hello,\n\n"
            "I noticed that I was charged twice for my subscription this month — "
            "once on March 1st ($49.99) and again on March 8th ($49.99). "
            "My account is jane.doe@example.com. This is really frustrating as I've "
            "been a loyal customer for 2 years. I need this resolved urgently and "
            "would appreciate a refund for the duplicate charge.\n\n"
            "Best,\nJane Doe"
        ),
        "sender": "jane.doe@example.com",
        "timestamp": "2024-03-15T10:30:00Z",
    },
    {
        "id": "r002",
        "subject": "Wrong item shipped — order #78423",
        "body": (
            "Hi support,\n\n"
            "I received my order #78423 today but the item inside was completely "
            "wrong — I ordered a blue 16GB USB drive and received a red mouse "
            "instead. I need the correct item urgently for a work project due "
            "Friday. Please arrange an immediate replacement and a return label "
            "for the wrong item. My account email is tom.baker@company.org.\n\n"
            "Regards,\nTom Baker"
        ),
        "sender": "tom.baker@company.org",
        "timestamp": "2024-03-16T08:15:00Z",
    },
    {
        "id": "r003",
        "subject": "Account locked — cannot access my data",
        "body": (
            "Hello,\n\n"
            "My account has been locked for the past 48 hours and I cannot access "
            "any of my stored files. I have an important presentation tomorrow and "
            "all my slides are in your cloud storage. I have not violated any terms "
            "of service. My account is sarah.jones@startup.io. Please unlock it "
            "immediately or at least tell me what happened. This is unacceptable.\n\n"
            "Sarah Jones"
        ),
        "sender": "sarah.jones@startup.io",
        "timestamp": "2024-03-17T14:45:00Z",
    },
]

# Backwards-compatible single-email alias (index 0)
RESPONSE_EMAIL: Dict = RESPONSE_EMAILS[0]

# ---------------------------------------------------------------------------
# Response grading criteria — shared across all response email fixtures.
# The criterion key "mentions_resolution" is intentionally generic so it
# applies to refunds (r001), replacements (r002), and account unlocks (r003).
# Per-email resolution patterns live in graders.py.
# ---------------------------------------------------------------------------
RESPONSE_CRITERIA = {
    "has_greeting": {
        "description": "Response opens with a greeting addressing the customer",
        "weight": 0.10,
    },
    "acknowledges_issue": {
        "description": "Acknowledges the specific complaint raised by the customer",
        "weight": 0.20,
    },
    "apologizes": {
        "description": "Contains an apology for the inconvenience",
        "weight": 0.15,
    },
    "mentions_resolution": {
        "description": "Mentions processing, investigating, or arranging a resolution (refund, replacement, unlock, etc.)",
        "weight": 0.25,
    },
    "requests_verification": {
        "description": "Asks for or confirms account/order details for verification",
        "weight": 0.10,
    },
    "professional_tone": {
        "description": "Maintains professional, empathetic tone throughout",
        "weight": 0.10,
    },
    "has_signoff": {
        "description": "Ends with a professional sign-off",
        "weight": 0.10,
    },
}