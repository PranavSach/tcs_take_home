"""
Setup script: Creates PostgreSQL tables, seeds synthetic data, and generates sample PDF documents.
Idempotent — safe to run multiple times. Drops and recreates tables on each run.
"""

import os
import random
from datetime import datetime, timedelta

import psycopg2
from faker import Faker
from fpdf import FPDF

from config import POSTGRES_URL

# Seeded for deterministic output
fake = Faker()
Faker.seed(42)
random.seed(42)

DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), "documents")


# ---------------------------------------------------------------------------
# Database Setup
# ---------------------------------------------------------------------------

DROP_TABLES = """
DROP TABLE IF EXISTS interactions CASCADE;
DROP TABLE IF EXISTS support_tickets CASCADE;
DROP TABLE IF EXISTS customers CASCADE;
"""

CREATE_CUSTOMERS = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id SERIAL PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    phone VARCHAR(20),
    plan VARCHAR(20) CHECK (plan IN ('free', 'basic', 'premium', 'enterprise')),
    signup_date DATE NOT NULL,
    status VARCHAR(20) CHECK (status IN ('active', 'inactive', 'suspended')) DEFAULT 'active'
);
"""

CREATE_TICKETS = """
CREATE TABLE IF NOT EXISTS support_tickets (
    ticket_id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(customer_id),
    subject VARCHAR(200) NOT NULL,
    description TEXT,
    status VARCHAR(20) CHECK (status IN ('open', 'in_progress', 'resolved', 'closed')) DEFAULT 'open',
    priority VARCHAR(10) CHECK (priority IN ('low', 'medium', 'high', 'critical')),
    category VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP
);
"""

CREATE_INTERACTIONS = """
CREATE TABLE IF NOT EXISTS interactions (
    interaction_id SERIAL PRIMARY KEY,
    ticket_id INTEGER REFERENCES support_tickets(ticket_id),
    agent_name VARCHAR(100),
    message TEXT NOT NULL,
    sender VARCHAR(20) CHECK (sender IN ('customer', 'agent', 'system')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def get_connection():
    """Create a psycopg2 connection from the configured POSTGRES_URL."""
    return psycopg2.connect(POSTGRES_URL)


def create_tables(conn):
    """Drop existing tables and recreate them."""
    with conn.cursor() as cur:
        cur.execute(DROP_TABLES)
        cur.execute(CREATE_CUSTOMERS)
        cur.execute(CREATE_TICKETS)
        cur.execute(CREATE_INTERACTIONS)
    conn.commit()
    print("Tables created successfully.")


# ---------------------------------------------------------------------------
# Seed Data
# ---------------------------------------------------------------------------

PLANS = ["free", "basic", "premium", "enterprise"]
STATUSES = ["active", "inactive", "suspended"]
TICKET_STATUSES = ["open", "in_progress", "resolved", "closed"]
PRIORITIES = ["low", "medium", "high", "critical"]
CATEGORIES = [
    "billing", "technical", "account", "feature_request",
    "bug_report", "general_inquiry", "security", "onboarding"
]

TICKET_SUBJECTS = [
    "Cannot access premium features",
    "Billing discrepancy on last invoice",
    "Request for data export",
    "App crashes on login",
    "Password reset not working",
    "Unable to upgrade plan",
    "Missing transaction records",
    "API rate limit exceeded",
    "Account locked after failed attempts",
    "Feature request: dark mode",
    "Integration with Slack not working",
    "Slow dashboard loading times",
    "Cannot add team members",
    "Invoice shows wrong amount",
    "Two-factor authentication issue",
    "Data not syncing across devices",
    "Need help with API documentation",
    "Subscription cancellation request",
    "Custom report generation failing",
    "Email notifications not received",
    "Mobile app not loading",
    "Permission denied on shared folder",
    "Webhook delivery failures",
    "SSO configuration help needed",
    "Export format not supported",
    "Search function returning wrong results",
    "Calendar sync broken",
    "File upload size limit too small",
    "Automated workflow stopped running",
    "Need bulk user import feature",
]

AGENT_NAMES = ["Sarah Mitchell", "David Park", "Maria Garcia", "James Wilson", "Lisa Chen"]

CUSTOMER_MESSAGES = [
    "Hi, I'm experiencing an issue with {subject}. Can you help?",
    "This has been happening since yesterday. {subject} is really affecting my work.",
    "I've tried restarting and clearing cache but the problem persists.",
    "Thank you for looking into this. Let me know if you need more details.",
    "Is there an estimated time for resolution?",
    "I noticed the issue started after the last update.",
    "This is urgent as it's blocking my team's workflow.",
    "Could you please escalate this? We have a deadline coming up.",
]

AGENT_MESSAGES = [
    "Thank you for reaching out. I'm looking into this right now.",
    "I've identified the issue and our engineering team is working on a fix.",
    "Could you please provide your account email so I can investigate further?",
    "I've applied a temporary fix. Please try again and let me know if the issue persists.",
    "This has been escalated to our senior engineering team.",
    "The fix has been deployed. You should see the resolution within the next few minutes.",
    "I've updated your account settings which should resolve the issue.",
    "Our team has identified the root cause and a permanent fix is in progress.",
]

SYSTEM_MESSAGES = [
    "Ticket created and assigned to support queue.",
    "Ticket priority updated to {priority}.",
    "Ticket status changed to {status}.",
    "Customer satisfaction survey sent.",
]


def seed_customers(conn):
    """Insert 20 customers including Ema Johnson."""
    customers = []

    # Ensure Ema Johnson is first
    customers.append((
        "Ema", "Johnson", "ema.johnson@email.com", "+1-555-0101",
        "premium", datetime(2023, 3, 15).date(), "active"
    ))

    # Generate 19 more customers
    used_emails = {"ema.johnson@email.com"}
    for _ in range(19):
        first = fake.first_name()
        last = fake.last_name()
        email = f"{first.lower()}.{last.lower()}@{fake.free_email_domain()}"
        while email in used_emails:
            email = f"{first.lower()}.{last.lower()}{random.randint(1, 99)}@{fake.free_email_domain()}"
        used_emails.add(email)
        phone = fake.phone_number()[:20]
        plan = random.choice(PLANS)
        signup = fake.date_between(start_date="-3y", end_date="-30d")
        status = random.choices(STATUSES, weights=[70, 20, 10])[0]
        customers.append((first, last, email, phone, plan, signup, status))

    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO customers (first_name, last_name, email, phone, plan, signup_date, status)
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            customers
        )
    conn.commit()
    print(f"Seeded {len(customers)} customers.")
    return len(customers)


def seed_tickets(conn, num_customers):
    """Insert 50 support tickets. Ema (customer_id=1) gets 4 tickets."""
    tickets = []

    # 4 tickets for Ema Johnson (customer_id=1)
    ema_subjects = [
        ("Cannot access premium features", "I upgraded to premium last week but still can't access premium dashboards.", "open", "high", "technical"),
        ("Billing discrepancy on last invoice", "My last invoice shows $99 but my plan should be $79/month.", "in_progress", "medium", "billing"),
        ("Request for data export", "I need to export all my project data in CSV format for compliance.", "resolved", "low", "feature_request"),
        ("Email notifications not received", "I stopped receiving email alerts for ticket updates since Monday.", "open", "medium", "technical"),
    ]
    base_date = datetime(2024, 1, 10, 9, 0, 0)
    for i, (subj, desc, status, priority, category) in enumerate(ema_subjects):
        created = base_date + timedelta(days=i * 15, hours=random.randint(0, 8))
        resolved = created + timedelta(days=random.randint(1, 5)) if status in ("resolved", "closed") else None
        tickets.append((1, subj, desc, status, priority, category, created, resolved))

    # 46 tickets for other customers
    for _ in range(46):
        cust_id = random.randint(1, num_customers)
        subj = random.choice(TICKET_SUBJECTS)
        desc = f"Customer reported: {subj.lower()}. This needs to be investigated and resolved."
        status = random.choice(TICKET_STATUSES)
        priority = random.choices(PRIORITIES, weights=[30, 35, 25, 10])[0]
        category = random.choice(CATEGORIES)
        created = fake.date_time_between(start_date="-6M", end_date="-1d")
        resolved = created + timedelta(days=random.randint(1, 14)) if status in ("resolved", "closed") else None
        tickets.append((cust_id, subj, desc, status, priority, category, created, resolved))

    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO support_tickets
               (customer_id, subject, description, status, priority, category, created_at, resolved_at)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
            tickets
        )
    conn.commit()
    print(f"Seeded {len(tickets)} support tickets.")
    return len(tickets)


def seed_interactions(conn, num_tickets):
    """Insert ~120 interactions across all tickets (2-4 per ticket)."""
    interactions = []

    for ticket_id in range(1, num_tickets + 1):
        num_msgs = random.randint(2, 4)
        agent = random.choice(AGENT_NAMES)
        base_time = fake.date_time_between(start_date="-6M", end_date="-1d")

        for j in range(num_msgs):
            created = base_time + timedelta(hours=j * random.randint(1, 12))

            if j == 0:
                # First message is always from customer
                msg = random.choice(CUSTOMER_MESSAGES).format(subject="this issue")
                sender = "customer"
                agent_name = None
            elif j == num_msgs - 1 and random.random() < 0.3:
                # Occasionally end with a system message
                msg = random.choice(SYSTEM_MESSAGES).format(
                    priority=random.choice(PRIORITIES),
                    status=random.choice(TICKET_STATUSES)
                )
                sender = "system"
                agent_name = None
            elif j % 2 == 1:
                msg = random.choice(AGENT_MESSAGES)
                sender = "agent"
                agent_name = agent
            else:
                msg = random.choice(CUSTOMER_MESSAGES).format(subject="my issue")
                sender = "customer"
                agent_name = None

            interactions.append((ticket_id, agent_name, msg, sender, created))

    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO interactions (ticket_id, agent_name, message, sender, created_at)
               VALUES (%s, %s, %s, %s, %s)""",
            interactions
        )
    conn.commit()
    print(f"Seeded {len(interactions)} interactions.")


# ---------------------------------------------------------------------------
# PDF Generation
# ---------------------------------------------------------------------------

class PolicyPDF(FPDF):
    """Custom PDF class with consistent header/footer styling."""

    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, "TechCorp Solutions - Confidential", align="R", new_x="LMARGIN", new_y="NEXT")
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def chapter_title(self, title):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(0, 51, 102)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def chapter_body(self, text):
        self.set_font("Helvetica", "", 11)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 6, text)
        self.ln(4)

    def section_header(self, title):
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(0, 51, 102)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)


def generate_refund_policy_pdf():
    """Generate company_refund_policy.pdf with realistic content."""
    pdf = PolicyPDF()
    pdf.alias_nb_pages()
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 15, "TechCorp Solutions", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Refund Policy", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, "Effective Date: January 1, 2024 | Last Updated: March 1, 2024", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)

    # Section 1: Refund Eligibility
    pdf.chapter_title("1. Refund Eligibility")
    pdf.chapter_body(
        "TechCorp Solutions is committed to customer satisfaction. Our refund policy is designed to be fair "
        "and transparent for all subscription tiers.\n\n"
        "Full Refund (within 30 days): Customers who cancel their subscription within 30 days of the initial "
        "purchase or renewal date are eligible for a full refund of the most recent billing cycle. This applies "
        "to all subscription plans including Basic ($29/month), Premium ($79/month), and Enterprise (custom pricing).\n\n"
        "Partial Refund (31-60 days): Customers who cancel between 31 and 60 days from the billing date may "
        "receive a pro-rated refund calculated based on the unused portion of their subscription period. The "
        "refund amount is calculated as: (Remaining Days / Total Days in Billing Cycle) x Monthly Fee.\n\n"
        "No Refund (after 60 days): Cancellations made after 60 days from the billing date are not eligible for "
        "refunds. However, the subscription will remain active until the end of the current billing cycle."
    )

    # Section 2: Refund Process
    pdf.chapter_title("2. Refund Process")
    pdf.chapter_body(
        "To request a refund, customers should follow these steps:\n\n"
        "Step 1: Log into your TechCorp Solutions account and navigate to Settings > Billing > Request Refund.\n\n"
        "Step 2: Fill out the refund request form, providing your reason for cancellation and any relevant details.\n\n"
        "Step 3: Submit the request. You will receive a confirmation email within 24 hours.\n\n"
        "Step 4: Our billing team will review the request within 3-5 business days.\n\n"
        "Step 5: If approved, the refund will be processed to the original payment method within 5-10 business "
        "days from the approval date.\n\n"
        "Customers may also contact our support team at support@techcorpsolutions.com or call 1-800-TECHCORP "
        "to initiate a refund request."
    )

    pdf.add_page()

    # Section 3: Exceptions
    pdf.chapter_title("3. Exceptions and Special Conditions")
    pdf.chapter_body(
        "The following situations have specific refund conditions:\n\n"
        "Digital Products and Add-ons: One-time purchases of digital products, integrations, or premium add-ons "
        "are non-refundable after activation. Customers have a 48-hour window after purchase to request a refund "
        "for unused digital products.\n\n"
        "Enterprise Contracts: Enterprise customers with annual contracts are subject to the terms outlined in "
        "their individual service agreements. Standard refund timelines may not apply. Please contact your "
        "designated account manager for enterprise refund inquiries.\n\n"
        "Promotional Subscriptions: Subscriptions purchased during promotional events (e.g., Black Friday, annual "
        "sales) at discounted rates are eligible for refund based on the discounted price paid, not the standard "
        "retail price.\n\n"
        "Free Trial Conversions: If a customer's free trial automatically converts to a paid subscription, they "
        "have 14 days from the conversion date to request a full refund, regardless of the standard 30-day policy.\n\n"
        "Service Outages: In the event of extended service outages exceeding 24 hours, affected customers are "
        "eligible for service credits proportional to the downtime experienced, regardless of the refund window."
    )

    # Section 4: Escalation
    pdf.chapter_title("4. Escalation Procedures")
    pdf.chapter_body(
        "If a refund request is denied and you believe the decision is incorrect, you may escalate through "
        "the following channels:\n\n"
        "Level 1 - Support Manager: Reply to the denial email or request to speak with a support manager. "
        "The manager will review the case within 2 business days.\n\n"
        "Level 2 - Customer Advocacy Team: If the Level 1 review does not resolve the issue, contact our "
        "Customer Advocacy Team at advocacy@techcorpsolutions.com. They will conduct an independent review "
        "within 5 business days.\n\n"
        "Level 3 - Executive Review: As a final step, customers may request an executive review by writing to "
        "executive-review@techcorpsolutions.com. This review is conducted by a member of the senior leadership "
        "team and a response will be provided within 10 business days.\n\n"
        "All escalation decisions are final and binding."
    )

    path = os.path.join(DOCUMENTS_DIR, "company_refund_policy.pdf")
    pdf.output(path)
    print(f"Generated: {path}")


def generate_terms_of_service_pdf():
    """Generate terms_of_service.pdf with realistic content."""
    pdf = PolicyPDF()
    pdf.alias_nb_pages()
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 15, "TechCorp Solutions", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Terms of Service", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, "Effective Date: January 1, 2024 | Version 3.2", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)

    # Section 1: Account Terms
    pdf.chapter_title("1. Account Terms")
    pdf.chapter_body(
        "By creating an account with TechCorp Solutions, you agree to the following terms:\n\n"
        "Age Requirement: You must be at least 18 years old (or the age of majority in your jurisdiction) to "
        "create an account and use our services. Users between 13 and 18 may use the service with verified "
        "parental or guardian consent.\n\n"
        "Account Responsibility: You are responsible for maintaining the security of your account credentials, "
        "including your password and any API keys. TechCorp Solutions is not liable for any loss or damage "
        "arising from unauthorized access to your account due to your failure to safeguard your credentials.\n\n"
        "Accurate Information: You must provide accurate, current, and complete information during registration "
        "and keep your account information updated. Providing false information is grounds for immediate "
        "account termination.\n\n"
        "One Account Per Person: Each individual may maintain only one account. Creating multiple accounts to "
        "circumvent restrictions or abuse free tier limitations will result in account suspension."
    )

    # Section 2: Acceptable Use
    pdf.chapter_title("2. Acceptable Use Policy")
    pdf.chapter_body(
        "You agree not to use TechCorp Solutions services for any of the following prohibited activities:\n\n"
        "- Unauthorized access to other users' accounts or data\n"
        "- Distributing malware, viruses, or any harmful code through our platform\n"
        "- Using automated scripts or bots to scrape data or overload our systems\n"
        "- Storing or transmitting content that violates applicable laws or regulations\n"
        "- Reselling or redistributing access to our services without authorization\n"
        "- Attempting to reverse engineer, decompile, or disassemble our software\n"
        "- Using our platform to send unsolicited communications or spam\n"
        "- Engaging in any activity that disrupts or interferes with our services\n\n"
        "Violation of this policy may result in immediate account suspension or termination, at TechCorp "
        "Solutions' sole discretion."
    )

    pdf.add_page()

    # Section 3: Data Privacy
    pdf.chapter_title("3. Data Privacy and Security")
    pdf.chapter_body(
        "TechCorp Solutions takes data privacy seriously and is committed to protecting your information:\n\n"
        "Data Collection: We collect only the data necessary to provide our services, including account "
        "information, usage data, and support interactions. We do not sell personal data to third parties.\n\n"
        "Data Storage: All customer data is stored in encrypted form using AES-256 encryption at rest and "
        "TLS 1.3 for data in transit. Our data centers are located in the United States and European Union, "
        "compliant with SOC 2 Type II and GDPR requirements.\n\n"
        "Data Retention: Customer data is retained for the duration of the account plus 90 days after "
        "account closure. After this period, data is permanently deleted from our systems.\n\n"
        "Data Portability: Customers may export their data at any time through the Settings > Data Export "
        "feature. We support exports in CSV, JSON, and XML formats.\n\n"
        "Breach Notification: In the event of a data breach, affected customers will be notified within 72 "
        "hours via email and in-app notification, in compliance with applicable regulations."
    )

    # Section 4: SLA
    pdf.chapter_title("4. Service Level Agreement (SLA)")
    pdf.chapter_body(
        "TechCorp Solutions commits to the following service levels:\n\n"
        "Uptime Guarantee: We guarantee 99.9% uptime for all paid subscription tiers, measured on a monthly "
        "basis. This excludes planned maintenance windows, which are scheduled during off-peak hours "
        "(Saturday 2:00 AM - 6:00 AM UTC) with at least 48 hours advance notice.\n\n"
        "Support Response Times:\n"
        "- Critical issues (service down): Response within 1 hour, resolution target within 4 hours\n"
        "- High priority issues: Response within 4 hours, resolution target within 24 hours\n"
        "- Medium priority issues: Response within 8 hours, resolution target within 48 hours\n"
        "- Low priority issues: Response within 24 hours, resolution target within 5 business days\n\n"
        "Service Credits: If monthly uptime falls below 99.9%, customers are eligible for service credits:\n"
        "- 99.0% - 99.9% uptime: 10% credit on monthly bill\n"
        "- 95.0% - 99.0% uptime: 25% credit on monthly bill\n"
        "- Below 95.0% uptime: 50% credit on monthly bill\n\n"
        "Service credits must be requested within 30 days of the incident and are applied to the next "
        "billing cycle."
    )

    pdf.add_page()

    # Section 5: Termination
    pdf.chapter_title("5. Termination")
    pdf.chapter_body(
        "Account termination may occur under the following conditions:\n\n"
        "Voluntary Termination: Customers may terminate their account at any time through Settings > Account > "
        "Close Account. Upon termination, access to services will continue until the end of the current billing "
        "cycle. Data will be retained for 90 days after termination, after which it will be permanently deleted.\n\n"
        "Termination for Cause: TechCorp Solutions reserves the right to suspend or terminate accounts that:\n"
        "- Violate the Acceptable Use Policy\n"
        "- Fail to pay subscription fees after 30 days past due\n"
        "- Engage in fraudulent activity\n"
        "- Are subject to a valid legal order requiring account closure\n\n"
        "In cases of termination for cause, TechCorp Solutions may immediately restrict access to the account "
        "and services. The customer will be notified via email and given 30 days to export their data before "
        "permanent deletion.\n\n"
        "Effect of Termination: Upon termination, all licenses and rights granted under these terms will "
        "immediately cease. Any provisions that by their nature should survive termination (including but not "
        "limited to confidentiality obligations, limitation of liability, and dispute resolution) will continue "
        "to apply."
    )

    # Section 6: Contact
    pdf.chapter_title("6. Contact Information")
    pdf.chapter_body(
        "For questions regarding these Terms of Service, please contact:\n\n"
        "TechCorp Solutions, Inc.\n"
        "123 Innovation Drive, Suite 400\n"
        "San Francisco, CA 94105\n"
        "Email: legal@techcorpsolutions.com\n"
        "Phone: 1-800-TECHCORP\n\n"
        "For support inquiries: support@techcorpsolutions.com"
    )

    path = os.path.join(DOCUMENTS_DIR, "terms_of_service.pdf")
    pdf.output(path)
    print(f"Generated: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(DOCUMENTS_DIR, exist_ok=True)

    # Generate PDFs (deterministic, no DB needed)
    print("\n--- Generating PDF Documents ---")
    generate_refund_policy_pdf()
    generate_terms_of_service_pdf()

    # Database setup
    print("\n--- Setting Up Database ---")
    try:
        conn = get_connection()
    except Exception as e:
        print(f"ERROR: Could not connect to PostgreSQL: {e}")
        print("Make sure PostgreSQL is running: docker-compose up -d")
        return

    try:
        create_tables(conn)
        num_customers = seed_customers(conn)
        num_tickets = seed_tickets(conn, num_customers)
        seed_interactions(conn, num_tickets)
        print("\nSetup complete! Database seeded and PDFs generated.")
    except Exception as e:
        conn.rollback()
        print(f"ERROR during seeding: {e}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
