"""Synthetic multi-page documents for demos, tests and benchmarks.

Provides a realistic annual-report-style financial filing and a services
agreement, each with explicit page markers (``=== PAGE N ===``) and section
headers, embedded figures and risk language — so the analyst has real, citable
answers to reason across pages.
"""

from __future__ import annotations

from .models import Document

ANNUAL_REPORT = """\
=== PAGE 1 ===
Northwind Robotics, Inc.
Annual Report and Form 10-K
Fiscal Year 2025

This report contains forward-looking statements about the company's operations,
financial performance, and strategy.

=== PAGE 2 ===
## Financial Highlights
Total revenue for fiscal year 2025 was $482.6 million, compared to $401.2 million
in fiscal 2024. Quarterly revenue was $101.4 million in Q1, $112.8 million in Q2,
$138.9 million in Q3, and $129.5 million in Q4. Gross margin improved to 58.2%
from 54.1% in the prior year. Net income was $44.3 million.

## Cash Position
The company ended the year with $210.0 million in cash and cash equivalents and
no long-term debt.

=== PAGE 3 ===
## Revenue Analysis
Third quarter revenue grew to $138.9 million from $112.8 million in the second
quarter, driven by strong demand for the Atlas autonomous platform. Year over
year, Q3 revenue increased from $96.4 million in the prior-year quarter. The
Services segment contributed $41.2 million of Q3 revenue, up 33% year over year.

## Guidance
Management expects full-year 2026 revenue between $560 million and $585 million.

=== PAGE 4 ===
## Risk Factors
The company faces several primary risk factors. Supply chain exposure to a single
semiconductor vendor could materially disrupt production and increase costs.
Product liability claims arising from autonomous systems could result in
significant legal expense. Foreign currency exposure affects roughly 30% of
revenue denominated in euros and yen. Cybersecurity incidents could compromise
customer data and trigger regulatory penalties. Intense competition may erode
pricing and margins.

=== PAGE 5 ===
## Legal Proceedings
The company is a defendant in a patent infringement lawsuit filed in March 2025.
Management believes the claims are without merit but an adverse outcome could
result in damages of up to $25 million.

## Employees
As of year end the company employed 2,140 full-time staff across 6 countries.
"""

SERVICES_AGREEMENT = """\
=== PAGE 1 ===
MASTER SERVICES AGREEMENT
Between Northwind Robotics, Inc. ("Provider") and Orion Logistics, LLC ("Client")
Effective Date: January 15, 2025

=== PAGE 2 ===
## Term and Termination
This Agreement shall commence on the Effective Date and continue for an initial
term of 36 months. Either party may terminate for material breach upon 30 days
written notice if the breach remains uncured.

## Fees
The Client shall pay Provider a fee of $75,000 per month. Late payments accrue
interest at 1.5% per month.

=== PAGE 3 ===
## Limitation of Liability
Provider's total liability under this Agreement shall not exceed the total fees
paid by Client in the 12 months preceding the claim. In no event shall either
party be liable for indirect or consequential damages.

## Indemnification
Each party shall indemnify the other against third-party claims arising from its
gross negligence or willful misconduct.

=== PAGE 4 ===
## Confidentiality
Each party shall protect the other's confidential information for a period of 5
years following termination. Data breaches must be reported within 72 hours.
"""


def sample_documents() -> list[Document]:
    return [
        Document(title="Northwind Robotics FY2025 Annual Report", text=ANNUAL_REPORT,
                 source="annual_report.txt"),
        Document(title="Master Services Agreement", text=SERVICES_AGREEMENT,
                 source="services_agreement.txt"),
    ]


SAMPLE_QUESTIONS = [
    "What was the total revenue growth in Q3, and what are the primary risk factors?",
    "What is the limitation of liability in the services agreement?",
    "How much cash did the company end the year with?",
    "What is the monthly fee under the services agreement?",
    "What is the company's guidance for 2026?",
    "Who is the current CEO and what is their salary?",   # not in the docs -> should refuse
]
