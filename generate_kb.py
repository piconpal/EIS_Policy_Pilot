"""
generate_kb.py - Synthetic Knowledge Base PDF Generator for Enterprise RAG
Generates 35 domain-coherent security PDFs using reportlab.
Run: python generate_kb.py

After PDFs are generated, this script attempts to clear the in-memory query
cache on the running RAG server (POST /cache/clear on localhost:8000) so that
stale retrieval results are not served after the knowledge base is updated.
If the server is not running, the cache will be cleared automatically on the
next server startup.
"""

import urllib.request
import urllib.error
from pathlib import Path
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle,
    HRFlowable, ListFlowable, ListItem,
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.graphics.shapes import Drawing, String
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.legends import Legend

OUTPUT_DIR = Path("data/raw")

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

def build_styles():
    return {
        "title": ParagraphStyle(
            "DocTitle", fontSize=28, leading=36, alignment=TA_CENTER,
            spaceAfter=20, fontName="Helvetica-Bold", textColor=colors.HexColor("#1a1a2e")
        ),
        "subtitle": ParagraphStyle(
            "DocSubtitle", fontSize=16, leading=22, alignment=TA_CENTER,
            spaceAfter=14, fontName="Helvetica", textColor=colors.HexColor("#16213e")
        ),
        "meta": ParagraphStyle(
            "DocMeta", fontSize=11, leading=16, alignment=TA_CENTER,
            spaceAfter=8, fontName="Helvetica", textColor=colors.grey
        ),
        "h1": ParagraphStyle(
            "H1", fontSize=20, leading=26, spaceBefore=24, spaceAfter=10,
            fontName="Helvetica-Bold", textColor=colors.HexColor("#0f3460")
        ),
        "h2": ParagraphStyle(
            "H2", fontSize=15, leading=20, spaceBefore=16, spaceAfter=8,
            fontName="Helvetica-Bold", textColor=colors.HexColor("#1a1a2e")
        ),
        "h3": ParagraphStyle(
            "H3", fontSize=12, leading=16, spaceBefore=10, spaceAfter=6,
            fontName="Helvetica-Bold", textColor=colors.HexColor("#16213e")
        ),
        "body": ParagraphStyle(
            "Body", fontSize=10, leading=15, spaceAfter=8,
            fontName="Helvetica", alignment=TA_JUSTIFY
        ),
        "toc_ch": ParagraphStyle(
            "TOCChapter", fontSize=11, leading=16, spaceAfter=4,
            fontName="Helvetica-Bold"
        ),
        "toc_sec": ParagraphStyle(
            "TOCSec", fontSize=10, leading=14, spaceAfter=2,
            fontName="Helvetica", leftIndent=20
        ),
        "caption": ParagraphStyle(
            "Caption", fontSize=9, leading=12, spaceAfter=6,
            fontName="Helvetica-Oblique", textColor=colors.grey, alignment=TA_CENTER
        ),
        "note": ParagraphStyle(
            "Note", fontSize=9, leading=13, spaceAfter=6,
            fontName="Helvetica-Oblique", textColor=colors.HexColor("#555555"),
            leftIndent=12
        ),
    }

# ---------------------------------------------------------------------------
# Reusable flowable builders (all accept styles dict)
# ---------------------------------------------------------------------------

def title_page(S, title, subtitle, doc_id, version="v2.1", classification="INTERNAL"):
    elems = [Spacer(1, 1.5 * inch)]
    elems.append(Paragraph(title, S["title"]))
    elems.append(Spacer(1, 0.3 * inch))
    elems.append(Paragraph(subtitle, S["subtitle"]))
    elems.append(Spacer(1, 0.2 * inch))
    elems.append(HRFlowable(width="80%", thickness=2, color=colors.HexColor("#0f3460"), spaceAfter=20))
    elems.append(Spacer(1, 0.3 * inch))
    for line in [
        f"Document ID: {doc_id}", f"Version: {version}",
        f"Classification: {classification}", "Owner: Information Security Architecture",
        "Review Cycle: Annual", "Last Reviewed: Q1 2026",
    ]:
        elems.append(Paragraph(line, S["meta"]))
    elems.append(Spacer(1, 0.5 * inch))
    elems.append(Paragraph(
        "This document contains proprietary and confidential information. "
        "Distribution is restricted to authorized personnel only.", S["note"]
    ))
    elems.append(PageBreak())
    return elems


def toc_page(S, chapters):
    """chapters: list of (num, title, [(sec_num, sec_title), ...])"""
    elems = [Paragraph("Table of Contents", S["h1"])]
    elems.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0f3460"), spaceAfter=14))
    for ch_num, ch_title, sections in chapters:
        elems.append(Paragraph(f"{ch_num}. {ch_title}", S["toc_ch"]))
        for sec_num, sec_title in sections:
            elems.append(Paragraph(f"    {ch_num}.{sec_num}  {sec_title}", S["toc_sec"]))
    elems.append(PageBreak())
    return elems


def chapter_header(S, num, title):
    elems = [HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey, spaceAfter=8)]
    elems.append(Paragraph(f"Chapter {num}", S["h3"]))
    elems.append(Paragraph(title, S["h1"]))
    elems.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0f3460"), spaceAfter=16))
    return elems


def section_block(S, sec_num, ch_num, title, body_elems):
    elems = [Paragraph(f"{ch_num}.{sec_num}  {title}", S["h2"])]
    elems.extend(body_elems)
    return elems


def kpi_table(S, caption, headers, rows):
    col_width = 6.5 * inch / len(headers)
    data = [headers] + rows
    tbl = Table(data, colWidths=[col_width] * len(headers), repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f3460")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4f8")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return [Spacer(1, 6), tbl, Paragraph(caption, S["caption"]), Spacer(1, 8)]


def numbered_list(S, items):
    li = [ListItem(Paragraph(item, S["body"]), value=i + 1) for i, item in enumerate(items)]
    return [ListFlowable(li, bulletType="1", start=1, leftIndent=20)]


def bullet_list(S, items):
    li = [ListItem(Paragraph(item, S["body"])) for item in items]
    return [ListFlowable(li, bulletType="bullet", leftIndent=20)]


# ---------------------------------------------------------------------------
# Chart helpers (reportlab.graphics)
# ---------------------------------------------------------------------------

_CHART_PALETTE = [
    colors.HexColor("#0f3460"), colors.HexColor("#e94560"),
    colors.HexColor("#4CAF50"), colors.HexColor("#FF9800"),
    colors.HexColor("#2196F3"), colors.HexColor("#9C27B0"),
    colors.HexColor("#00BCD4"), colors.HexColor("#FF5722"),
]


def bar_chart(S, caption, categories, series, width=450, height=220):
    """
    series: list of (label, [values]) tuples.
    Returns list of flowables: [Spacer, Drawing, caption Paragraph, Spacer].
    """
    d = Drawing(width, height)
    bc = VerticalBarChart()
    bc.x = 55
    bc.y = 45
    bc.width = width - 80
    bc.height = height - 75
    bc.data = [s[1] for s in series]
    bc.categoryAxis.categoryNames = categories
    bc.categoryAxis.labels.boxAnchor = "ne"
    bc.categoryAxis.labels.angle = 30 if len(categories) > 5 else 0
    bc.categoryAxis.labels.fontSize = 7
    bc.valueAxis.labels.fontSize = 7
    bc.valueAxis.forceZero = 1
    bc.groupSpacing = 8
    bc.barSpacing = 2
    for i, s in enumerate(series):
        bc.bars[i].fillColor = _CHART_PALETTE[i % len(_CHART_PALETTE)]
        bc.bars[i].strokeColor = colors.white
        bc.bars[i].strokeWidth = 0.5
    d.add(bc)
    if len(series) > 1:
        leg = Legend()
        leg.x = 55
        leg.y = 8
        leg.fontSize = 7
        leg.colorNamePairs = [(_CHART_PALETTE[i % len(_CHART_PALETTE)], s[0]) for i, s in enumerate(series)]
        leg.columnMaximum = 4
        leg.deltax = 75
        leg.deltay = 0
        leg.autoXPadding = 5
        d.add(leg)
    return [Spacer(1, 8), d, Paragraph(caption, S["caption"]), Spacer(1, 8)]


def line_chart(S, caption, categories, series, width=450, height=200):
    """
    series: list of (label, [values]) tuples.
    """
    d = Drawing(width, height)
    lc = HorizontalLineChart()
    lc.x = 55
    lc.y = 40
    lc.width = width - 80
    lc.height = height - 70
    lc.data = [s[1] for s in series]
    lc.categoryAxis.categoryNames = categories
    lc.categoryAxis.labels.fontSize = 7
    lc.valueAxis.labels.fontSize = 7
    lc.valueAxis.forceZero = 1
    for i, s in enumerate(series):
        lc.lines[i].strokeColor = _CHART_PALETTE[i % len(_CHART_PALETTE)]
        lc.lines[i].strokeWidth = 2
    d.add(lc)
    leg = Legend()
    leg.x = 55
    leg.y = 8
    leg.fontSize = 7
    leg.colorNamePairs = [(_CHART_PALETTE[i % len(_CHART_PALETTE)], s[0]) for i, s in enumerate(series)]
    leg.columnMaximum = 4
    leg.deltax = 100
    leg.deltay = 0
    d.add(leg)
    return [Spacer(1, 8), d, Paragraph(caption, S["caption"]), Spacer(1, 8)]


def pie_chart(S, caption, labels, data, width=420, height=230):
    """Pie chart flowable list."""
    d = Drawing(width, height)
    pie = Pie()
    pie.x = 60
    pie.y = 20
    pie.width = 170
    pie.height = 170
    pie.data = data
    pie.labels = labels
    pie.sideLabels = 1
    pie.sideLabelsOffset = 0.08
    pie.simpleLabels = 0
    pie.slices.strokeWidth = 0.8
    pie.slices.strokeColor = colors.white
    for i, c in enumerate(_CHART_PALETTE[:len(data)]):
        pie.slices[i].fillColor = c
    d.add(pie)
    return [Spacer(1, 8), d, Paragraph(caption, S["caption"]), Spacer(1, 8)]



def pad(S, topic, count=15):
    """Return substantive multi-sentence body paragraphs for RAG-quality content."""
    pool = [
        f"The {topic} framework is governed by a comprehensive set of enterprise-wide policies that define roles, responsibilities, and accountability structures across all business units and technology domains. These policies are reviewed annually by the Information Security Steering Committee and updated to reflect changes in the regulatory environment, threat landscape, and organizational risk appetite. Policy exceptions are tracked through the GRC platform and require formal CISO approval with documented compensating controls.",
        f"Compliance with {topic} requirements is measured on a quarterly basis through automated control testing, continuous monitoring, evidence collection, and management attestation processes embedded in the enterprise GRC platform. Control effectiveness ratings are assigned on a five-point scale from Ineffective to Highly Effective, with findings below Effective requiring a remediation plan within 30 days. Trending data is presented to the Board Risk Committee each quarter to facilitate informed oversight.",
        f"Risk appetite thresholds for {topic} are established by the CISO in collaboration with the Board Risk Committee and are reviewed annually to reflect changes in the threat landscape and organizational risk tolerance. Quantitative risk thresholds are expressed as maximum acceptable annualized loss expectancy (ALE) values per asset category, and qualitative thresholds are defined using a five-by-five risk matrix. Any risk finding that exceeds the defined appetite triggers a mandatory escalation to executive leadership within 24 hours.",
        f"Escalation procedures for {topic} violations must be initiated within two business hours of detection, with executive notification required for Severity-1 incidents affecting critical assets or regulated data. The escalation chain progresses from Security Analyst to SOC Manager, Security Director, CISO, and ultimately the CEO and Board if the incident meets the materiality threshold defined in the Incident Response Policy. Post-incident reviews are mandatory for all Severity-1 and Severity-2 events, with findings documented and tracked through the risk register.",
        f"The {topic} program integrates with the enterprise change management process to ensure that all new technology deployments, architecture changes, and configuration modifications are assessed for security impact prior to production release. Security review is a mandatory gate in the Change Advisory Board (CAB) approval workflow, with risk-tiered review requirements ranging from automated scanning for low-risk changes to full Security Architecture Review Board (SARB) evaluation for high-risk changes. Changes that introduce unacceptable risk are returned to the requestor with remediation guidance before approval.",
        f"Third-party assessments of {topic} controls are conducted annually by qualified external auditors and penetration testing firms, with findings classified by severity and tracked through the enterprise risk register until remediation is independently verified. The organization maintains a pre-approved vendor list for security assessment services, with providers evaluated against criteria including relevant certifications (CREST, OSCP, CISSP), independence requirements, and demonstrated industry expertise. Assessment reports and management responses are retained for a minimum of seven years to support regulatory audit requirements.",
        f"Training and awareness programs related to {topic} are mandatory for all personnel with access to sensitive systems or regulated data, with completion rates tracked through the Learning Management System (LMS) and reported to management monthly. Role-specific training modules provide targeted instruction for high-risk roles such as system administrators, developers, and finance personnel. Personnel who fail to complete mandatory training within the designated window have their system access suspended until training requirements are fulfilled.",
        f"Exceptions to {topic} policy requirements must be formally documented using the Policy Exception Request (PER) form, approved by the relevant system owner, Security Manager, and CISO, and are subject to mandatory compensating controls for the duration of the exception period. Exceptions are granted for a maximum of 90 days, after which they must be re-evaluated and either remediated or renewed with updated justification. All active exceptions are reviewed monthly by the Vulnerability Management and Risk teams to assess residual risk and accelerate remediation where possible.",
        f"Metrics and key performance indicators (KPIs) for {topic} are reported monthly to the Information Security Steering Committee and quarterly to the Board Audit and Risk Committee, with year-over-year trend analysis to demonstrate program maturity. Operational metrics include coverage percentages, mean time to detect, mean time to respond, and SLA adherence rates, while strategic metrics track risk reduction, cost per incident, and regulatory compliance posture. Dashboard visualizations are maintained in the enterprise GRC platform and accessible to senior stakeholders in real time.",
        f"Continuous improvement initiatives for the {topic} program are driven by a structured lessons-learned process following security incidents, outcomes from red team and purple team exercises, emerging threat intelligence, and annual benchmarking against industry peers using frameworks such as the NIST Cybersecurity Framework Maturity Model and CIS Controls Implementation Groups. Improvement initiatives are logged as projects in the security program roadmap, prioritized by risk reduction impact and resource requirements, and tracked through quarterly program reviews. Each initiative has a designated owner, success criteria, and target completion date.",
        f"The {topic} architecture follows a defense-in-depth model incorporating multiple complementary layers of preventive, detective, and corrective controls deployed across people, process, and technology dimensions. No single control is considered sufficient in isolation; rather, the layered approach ensures that the failure of any one control does not result in a complete security failure. Control interdependencies are documented in the enterprise control matrix, and compensating controls are identified for each primary control to maintain coverage during periods of maintenance or temporary unavailability.",
        f"Integration of {topic} capabilities with the enterprise SIEM enables real-time correlation of security events and automated response actions via SOAR playbooks, significantly reducing mean time to detect (MTTD) and mean time to respond (MTTR) compared to manual processes. Data feeds from {topic} systems are normalized to a common schema using vendor-neutral parsers, ensuring compatibility with the enterprise logging standard. Alert thresholds and correlation rules are reviewed monthly by detection engineers and tuned based on false positive rates and emerging threat intelligence.",
        f"The {topic} control framework is mapped to multiple regulatory standards including SOX, HIPAA, PCI DSS, GDPR, ISO/IEC 27001:2022, and NIST SP 800-53, enabling the organization to demonstrate compliance efficiently through a single set of evidence artifacts. Cross-framework mapping is maintained in the enterprise GRC platform, which automatically links control test results to relevant regulatory requirements, reducing duplication of effort during audit cycles. Internal audit independently validates control design and operating effectiveness on an annual basis, with findings reported directly to the Audit Committee.",
        f"The organization's {topic} maturity level is assessed annually using a five-level maturity model aligned to CMMI and the NIST CSF, with the current target state defined as Level 3 (Defined) or higher across all control domains. Maturity gaps identified during the assessment are incorporated into the security program roadmap as remediation initiatives with resource allocations and target completion dates. External benchmarking against industry peers in the same sector is conducted biannually to ensure the organization's maturity trajectory is competitive and aligned with evolving regulatory expectations.",
        f"Vendor and supply chain risks related to {topic} are managed through a structured third-party risk management (TPRM) program that requires all vendors with access to enterprise systems or data to complete an annual security questionnaire, maintain relevant certifications (ISO 27001, SOC 2 Type II), and undergo periodic on-site assessments for critical suppliers. Contractual security requirements are standardized in the Master Services Agreement (MSA) template and include the right to audit, incident notification obligations, and data handling standards. Vendor risk ratings are maintained in the TPRM platform and reviewed quarterly.",
        f"The {topic} program maintains a comprehensive evidence library that stores control test results, audit reports, assessment findings, exception approvals, and regulatory correspondence in a centralized, access-controlled repository with a retention period of seven years. Evidence is tagged to relevant controls, regulatory requirements, and time periods to facilitate rapid retrieval during internal and external audits. The evidence library is reviewed quarterly to ensure completeness, and gaps in evidence coverage are flagged for remediation by control owners.",
        f"Workforce planning for {topic} ensures that the security team maintains the skills and certifications required to operate and improve the program effectively. Required competencies are documented in job descriptions and assessed during annual performance reviews, with training plans developed for employees with identified skill gaps. The organization sponsors relevant professional certifications including CISSP, CISM, CRISC, CEH, and domain-specific credentials, and tracks certification status in the HR system to ensure continuous coverage of critical skill areas.",
        f"Technology investments supporting {topic} are evaluated through a security capability planning process that aligns tool selection with identified control gaps, threat intelligence priorities, and industry best practices. Tool evaluations include proof-of-concept testing, vendor financial stability assessments, integration compatibility reviews, and total cost of ownership analysis. Selected tools are onboarded through a structured implementation process that includes configuration baseline documentation, integration testing, operator training, and a 30-day hypercare period with vendor support.",
        f"The {topic} program charter is approved by the CISO and reviewed by the Board Risk Committee annually to ensure alignment with the organization's strategic objectives and risk appetite. The charter defines the program's mission, scope, authority, resource requirements, and key performance objectives, and serves as the foundational governance document for the program. Changes to the charter require executive approval and are communicated to all stakeholders through the enterprise policy management system.",
        f"Data governance requirements for {topic} ensure that all data processed, stored, or transmitted as part of the program is classified, labeled, and handled in accordance with the enterprise Data Classification Policy. Sensitive data elements including personally identifiable information (PII), protected health information (PHI), and payment card data (PCI) are subject to enhanced protection requirements including encryption at rest and in transit, access logging, and annual data minimization reviews. Data flows involving {topic} systems are documented in the enterprise data flow inventory and reviewed for compliance with applicable privacy regulations.",
    ]
    # Cycle through pool to fulfill count, avoiding indexing beyond pool size
    result = []
    for i in range(count):
        result.append(Paragraph(pool[i % len(pool)], S["body"]))
    return result


# ---------------------------------------------------------------------------
# Generic document assembler
# ---------------------------------------------------------------------------

def assemble_doc(S, title, subtitle, doc_id, chapters_def):
    """
    chapters_def: list of dicts with keys 'title' and 'sections'.
    Each section: {'title': str, 'body': list of flowables}
    """
    toc_chapters = [
        (i + 1, ch["title"], [(j + 1, s["title"]) for j, s in enumerate(ch["sections"])])
        for i, ch in enumerate(chapters_def)
    ]
    elems = title_page(S, title, subtitle, doc_id)
    elems += toc_page(S, toc_chapters)
    for i, ch in enumerate(chapters_def):
        elems += chapter_header(S, i + 1, ch["title"])
        for j, sec in enumerate(ch["sections"]):
            elems += section_block(S, j + 1, i + 1, sec["title"], sec["body"])
        elems.append(PageBreak())
    return elems

# ---------------------------------------------------------------------------
# IAM Documents
# ---------------------------------------------------------------------------

def doc_iam_rbac(S):
    title = "Role-Based Access Control (RBAC) Policy Framework"
    subtitle = "Enterprise Identity and Access Management"
    doc_id = "IAM-POL-001"
    chapters = [
        {"title": "Executive Overview", "sections": [
            {"title": "Purpose and Scope", "body": [
                Paragraph("This policy establishes the enterprise framework for Role-Based Access Control (RBAC) across all information systems, applications, and infrastructure components. RBAC restricts system access based on each user's role, enforcing the principle of least privilege across all on-premises, cloud, and hybrid environments.", S["body"]),
                Paragraph("The scope encompasses all employees, contractors, third-party vendors, and automated service accounts. It applies to all systems processing, storing, or transmitting sensitive or regulated data.", S["body"]),
                Paragraph("RBAC implementation must align with SOX Section 404, HIPAA §164.312(a)(1), PCI DSS Requirement 7, and NIST SP 800-53 AC-2, AC-3, and AC-6 control families.", S["body"]),
            ] + pad(S, "RBAC", 15)},
            {"title": "Regulatory Context", "body": [
                Paragraph("The regulatory landscape governing access control has expanded significantly. Frameworks such as ISO/IEC 27001:2022, NIST CSF 2.0, CIS Controls v8, and industry-specific mandates drive the need for robust, auditable RBAC implementations.", S["body"]),
                Paragraph("Organizations subject to SOX must demonstrate effective controls over financial system access, including the ability to detect and remediate segregation of duties (SoD) conflicts. PCI DSS merchants must restrict access to cardholder data based on business need-to-know.", S["body"]),
            ] + pad(S, "regulatory compliance", 15)},
            {"title": "Key Definitions", "body": bullet_list(S, [
                "Role: A named collection of permissions aligned to a job function or business process.",
                "Permission: An explicit authorization to perform an action on a resource.",
                "Principal: A user, group, or service account assigned to one or more roles.",
                "Segregation of Duties (SoD): Distributing tasks among multiple roles to prevent fraud.",
                "Role Mining: Discovering natural role groupings from existing access data.",
                "Entitlement: A specific access right granted through role assignment.",
            ])},
        ]},
        {"title": "RBAC Architecture", "sections": [
            {"title": "Role Hierarchy Model", "body": [
                Paragraph("The enterprise RBAC model is structured as a four-tier hierarchy: Base Roles, Functional Roles, Composite Roles, and Administrative Roles. Each tier inherits permissions from the tier below, with additional restrictions applied at higher tiers.", S["body"]),
                Paragraph("Functional Roles are aligned to specific job families such as Finance Analyst, Security Operations Analyst, or HR Business Partner. Composite Roles combine multiple functional roles for users with cross-functional responsibilities.", S["body"]),
            ] + pad(S, "role hierarchy", 15)},
            {"title": "Permission Sets", "body": kpi_table(S,
                "Table 2.1 – Standard Permission Set Categories",
                ["Permission Category", "Scope", "Approval Level", "Review Frequency"],
                [
                    ["Read", "Data retrieval only", "Manager", "Annual"],
                    ["Read/Write", "Data creation and modification", "Director", "Semi-annual"],
                    ["Privileged", "Admin or elevated access", "CISO", "Quarterly"],
                    ["Service Account", "Automated process access", "Architect + CISO", "Quarterly"],
                    ["Emergency/Break-glass", "Incident response access", "CISO + CTO", "Post-use"],
                ]
            )},
            {"title": "Segregation of Duties", "body": [
                Paragraph("SoD conflicts occur when a single role or user accumulates permissions that, in combination, create unacceptable risk of fraud, error, or policy violation. The enterprise maintains an SoD conflict matrix defining approximately 1,200 high-risk permission combinations across ERP, identity, financial, and infrastructure systems.", S["body"]),
                Paragraph("Automated SoD detection is performed continuously through the IGA platform, with violations surfaced as risk findings requiring management acknowledgment or remediation within defined SLA windows.", S["body"]),
            ] + pad(S, "SoD", 15)},
        ]},
        {"title": "Role Lifecycle Management", "sections": [
            {"title": "Role Creation Workflow", "body": numbered_list(S, [
                "Business unit submits Role Request Form (RRF) via the ITSM portal with justification, population size, and system scope.",
                "IAM team performs role mining analysis against existing entitlement data to identify overlap with existing roles.",
                "Security Architecture reviews the proposed permission set against the SoD conflict matrix.",
                "Risk Management evaluates any identified SoD conflicts and determines compensating controls.",
                "CISO approval required for roles containing privileged or sensitive permissions.",
                "Role is provisioned in IAM platform and documented in the Role Catalog.",
                "Notification sent to business owner confirming role availability.",
                "Role included in next quarterly access certification campaign.",
            ])},
            {"title": "Role Modification", "body": pad(S, "role modification", 15)},
            {"title": "Role Retirement", "body": pad(S, "role retirement", 15)},
        ]},
        {"title": "Access Assignment Procedures", "sections": [
            {"title": "Provisioning Process", "body": [
                Paragraph("Access provisioning follows a request-approve-fulfill workflow enforced through the Identity Governance and Administration (IGA) platform. All requests must originate from an authenticated user via self-service portal or ITSM integration. Automated provisioning is permitted for low-risk roles with pre-approved populations.", S["body"]),
            ] + pad(S, "provisioning", 15)},
            {"title": "Approval Chains", "body": kpi_table(S,
                "Table 4.1 – Access Approval Matrix by Risk Tier",
                ["Risk Tier", "Example Roles", "Approver 1", "Approver 2", "SLA"],
                [
                    ["Low", "Read-only, standard user", "Direct Manager", "—", "4 hours"],
                    ["Medium", "Write access, application admin", "Manager + App Owner", "—", "8 hours"],
                    ["High", "DBA, Infra Admin", "Manager + App Owner", "Security", "24 hours"],
                    ["Critical", "Domain Admin, CISO tools", "Manager + CISO", "CTO", "48 hours"],
                ]
            )},
            {"title": "Emergency Access", "body": pad(S, "emergency access", 15)},
        ]},
        {"title": "Privileged Role Controls", "sections": [
            {"title": "Admin Role Restrictions", "body": pad(S, "privileged role restrictions", 15)},
            {"title": "Just-In-Time Access", "body": [
                Paragraph("Just-In-Time (JIT) access controls restrict privileged role assignments to the minimum time window required. The enterprise PAM platform enforces JIT through time-bounded session checkout, with automatic revocation upon session expiry.", S["body"]),
                Paragraph("JIT sessions are fully recorded (keystrokes, screen capture, command logs) and stored for a minimum of 12 months in the immutable audit repository.", S["body"]),
            ] + pad(S, "JIT access", 15)},
            {"title": "Dual Control", "body": pad(S, "dual control", 15)},
        ]},
        {"title": "KPIs and Metrics", "sections": [
            {"title": "Role Coverage Metrics", "body": kpi_table(S,
                "Table 6.1 – RBAC KPI Dashboard",
                ["KPI", "Target", "Current", "Trend", "Owner"],
                [
                    ["Role Coverage (%)", "98%", "94.2%", "Improving", "IAM Team"],
                    ["Orphan Accounts", "<50", "23", "Stable", "IAM Team"],
                    ["SoD Violations Open", "<100", "67", "Declining", "Risk Mgmt"],
                    ["Avg Provisioning Time (hr)", "<8", "6.3", "Improving", "Service Desk"],
                    ["Access Cert Completion (%)", "100%", "96.8%", "Stable", "Business Owners"],
                    ["Privileged Accounts Reviewed", "100%", "98.1%", "Stable", "PAM Team"],
                ]
            ) + bar_chart(S,
                "Figure 6.1 – Role Coverage % by Business Unit (Q1 2026)",
                ["Finance", "HR", "IT", "Legal", "Operations", "Sales", "Security"],
                [("Coverage %", [91, 88, 99, 85, 93, 82, 100])],
            )},
            {"title": "SoD Violation Rates", "body": pad(S, "SoD metrics", 15)},
            {"title": "Access Review Completion", "body": pad(S, "access review metrics", 15)},
        ]},
        {"title": "Audit and Compliance", "sections": [
            {"title": "Audit Trail Requirements", "body": [
                Paragraph("All access provisioning, modification, and de-provisioning events must be logged with: timestamp (UTC), initiating principal, target account, role assigned/removed, approver identity, approval timestamp, and system of record. Log retention minimum is 7 years for systems subject to SOX and HIPAA.", S["body"]),
            ] + pad(S, "audit logging", 15)},
            {"title": "Reporting Standards", "body": pad(S, "audit reporting", 15)},
            {"title": "Regulatory Mapping", "body": kpi_table(S,
                "Table 7.1 – Regulatory Control Mapping",
                ["Regulation", "Control Requirement", "RBAC Control", "Evidence"],
                [
                    ["SOX 404", "Access to financial systems", "Role-based finance access", "Access cert reports"],
                    ["PCI DSS Req 7", "Need-to-know restriction", "Cardholder data roles", "Quarterly reviews"],
                    ["HIPAA §164.312", "Unique user ID", "Person-linked role assignment", "Audit logs"],
                    ["ISO 27001 A.9", "Access control policy", "RBAC policy documentation", "Policy register"],
                    ["NIST AC-6", "Least privilege", "Minimal role assignments", "Role catalog"],
                ]
            )},
        ]},
        {"title": "Governance and Review", "sections": [
            {"title": "Ownership Model", "body": pad(S, "governance ownership", 15)},
            {"title": "Policy Review Schedule", "body": pad(S, "policy review schedule", 15)},
            {"title": "Exception Management", "body": [
                Paragraph("All exceptions to RBAC policy must be submitted via the GRC platform using the Policy Exception Request (PER) form. Exceptions require CISO approval, a defined remediation timeline not exceeding 90 days, and implementation of documented compensating controls.", S["body"]),
            ] + pad(S, "exception management", 15)},
        ]},
    ]
    return assemble_doc(S, title, subtitle, doc_id, chapters)


def doc_iam_pam(S):
    title = "Privileged Access Management (PAM) Framework"
    subtitle = "Enterprise Identity and Access Management"
    doc_id = "IAM-POL-002"
    chapters = [
        {"title": "Introduction to Privileged Access Management", "sections": [
            {"title": "PAM Program Overview", "body": [
                Paragraph("Privileged Access Management (PAM) is a subset of identity and access management that focuses on monitoring, securing, and controlling elevated accounts that have greater-than-normal access to critical systems and sensitive data. These accounts represent the highest-risk identities within any enterprise environment.", S["body"]),
                Paragraph("The PAM program manages the full lifecycle of privileged accounts, enforces just-in-time access, records all privileged sessions, and detects anomalous behavior indicative of insider threat or external compromise.", S["body"]),
            ] + pad(S, "PAM", 15)},
            {"title": "Privileged Account Types", "body": bullet_list(S, [
                "Domain Administrator Accounts: Full control over Active Directory domain resources.",
                "Local Administrator Accounts: Administrative access on individual endpoints or servers.",
                "Database Administrator (DBA) Accounts: Full access to database management systems.",
                "Service Accounts: Non-human accounts used by applications and automated processes.",
                "Emergency/Break-glass Accounts: Last-resort accounts for disaster recovery scenarios.",
                "Cloud IAM Roles: Privileged cloud management roles (e.g., AWS AdministratorAccess, Azure Owner).",
                "Network Device Accounts: Admin access to routers, switches, and firewalls.",
            ])},
            {"title": "Regulatory Requirements", "body": pad(S, "PAM regulatory", 15)},
        ]},
        {"title": "PAM Architecture and Tooling", "sections": [
            {"title": "PAM Platform Components", "body": pad(S, "PAM architecture", 15)},
            {"title": "Credential Vaulting", "body": [
                Paragraph("All privileged credentials must be stored in the enterprise PAM vault. Credentials are rotated automatically by account risk tier: Critical accounts rotate every 24 hours, High every 7 days, Medium every 30 days.", S["body"]),
            ] + pad(S, "credential vaulting", 15)},
            {"title": "Session Recording and Monitoring", "body": pad(S, "session recording", 15)},
        ]},
        {"title": "Privileged Account Discovery", "sections": [
            {"title": "Automated Discovery Methods", "body": pad(S, "account discovery", 15)},
            {"title": "Rogue Account Detection", "body": pad(S, "rogue accounts", 15)},
            {"title": "Discovery Metrics", "body": kpi_table(S,
                "Table 3.1 – Privileged Account Discovery KPIs",
                ["Metric", "Target", "Q1 2026", "Status"],
                [
                    ["Accounts Under PAM Management", "100%", "97.3%", "On Track"],
                    ["Undiscovered Privileged Accounts", "0", "12", "Remediation In Progress"],
                    ["Stale Privileged Accounts (>90 days)", "<5", "3", "Compliant"],
                    ["Service Account Inventory Coverage", "100%", "94.1%", "In Progress"],
                ]
            )},
        ]},
        {"title": "Just-In-Time (JIT) Access Controls", "sections": [
            {"title": "JIT Workflow Design", "body": pad(S, "JIT workflow", 15)},
            {"title": "Time-Bounded Session Management", "body": pad(S, "session management", 15)},
            {"title": "Approval Automation", "body": pad(S, "approval automation", 15)},
        ]},
        {"title": "Privileged Session Management", "sections": [
            {"title": "Session Proxy Architecture", "body": pad(S, "session proxy", 15)},
            {"title": "Real-Time Monitoring and Alerting", "body": pad(S, "real-time monitoring", 15)},
            {"title": "Session Forensics", "body": pad(S, "session forensics", 15)},
        ]},
        {"title": "PAM KPIs and Reporting", "sections": [
            {"title": "Operational Metrics", "body": kpi_table(S,
                "Table 6.1 – PAM Operational KPI Dashboard",
                ["KPI", "Target", "Current", "Owner"],
                [
                    ["PAM Vault Coverage (%)", "100%", "97.3%", "PAM Team"],
                    ["Credential Rotation Compliance (%)", "100%", "99.1%", "PAM Team"],
                    ["Sessions Recorded (%)", "100%", "100%", "SOC"],
                    ["JIT Request SLA (hrs)", "<4", "2.1", "Service Desk"],
                    ["Anomalous Session Alerts (monthly)", "N/A", "14", "SOC"],
                    ["Privileged Account Cert Completion", "100%", "98.4%", "IAM Team"],
                ]
            )},
            {"title": "Executive Dashboard", "body": pad(S, "executive PAM reporting", 15)},
            {"title": "Regulatory Evidence Package", "body": pad(S, "regulatory evidence", 15)},
        ]},
        {"title": "Incident Response for PAM Events", "sections": [
            {"title": "PAM Incident Classification", "body": pad(S, "PAM incident classification", 15)},
            {"title": "Containment Procedures", "body": pad(S, "PAM containment", 15)},
            {"title": "Post-Incident Review", "body": pad(S, "post-incident review", 15)},
        ]},
        {"title": "Governance and Continuous Improvement", "sections": [
            {"title": "PAM Steering Committee", "body": pad(S, "PAM governance", 15)},
            {"title": "Maturity Assessment", "body": kpi_table(S,
                "Table 8.1 – PAM Maturity Model",
                ["Level", "Description", "Criteria", "Current Status"],
                [
                    ["1 - Initial", "Ad hoc privileged access", "No formal PAM program", "Completed"],
                    ["2 - Managed", "Basic vaulting in place", "Credential vault deployed", "Completed"],
                    ["3 - Defined", "JIT and session recording", "Full PAM platform active", "Current"],
                    ["4 - Measured", "KPI-driven optimization", "Dashboards and SLAs defined", "In Progress"],
                    ["5 - Optimized", "Predictive PAM analytics", "AI-driven anomaly detection", "Target 2027"],
                ]
            )},
            {"title": "Technology Roadmap", "body": pad(S, "PAM roadmap", 15)},
        ]},
    ]
    return assemble_doc(S, title, subtitle, doc_id, chapters)


def doc_iam_access_cert(S):
    title = "Access Certification and Recertification Policy"
    subtitle = "Enterprise Identity and Access Management"
    doc_id = "IAM-POL-003"
    chapters = [
        {"title": "Access Certification Overview", "sections": [
            {"title": "Policy Statement", "body": pad(S, "access certification policy", 15)},
            {"title": "Certification Types", "body": bullet_list(S, [
                "Annual User Access Review: All user entitlements reviewed by managers annually.",
                "Quarterly Privileged Access Review: Privileged accounts reviewed every 90 days.",
                "Event-Driven Certification: Triggered by role changes, transfers, or terminations.",
                "Application Onboarding Review: New application entitlements certified before go-live.",
                "SoD Conflict Review: Targeted review of users with SoD conflicts.",
            ])},
            {"title": "Roles and Responsibilities", "body": pad(S, "certification roles", 15)},
        ]},
        {"title": "Campaign Management", "sections": [
            {"title": "Campaign Planning", "body": pad(S, "campaign planning", 15)},
            {"title": "Reviewer Assignment", "body": pad(S, "reviewer assignment", 15)},
            {"title": "Campaign Timeline", "body": kpi_table(S,
                "Table 2.1 – Annual Certification Campaign Timeline",
                ["Phase", "Activity", "Duration", "Owner"],
                [
                    ["Preparation", "Data extraction and normalization", "Week 1", "IAM Team"],
                    ["Launch", "Reviewer notification and task assignment", "Day 1", "IAM Team"],
                    ["Review", "Manager certify/revoke decisions", "Days 1-14", "Managers"],
                    ["Escalation", "Overdue task escalation to skip-level", "Day 15", "IAM Team"],
                    ["Remediation", "Revocation of non-certified access", "Days 15-21", "IAM Team"],
                    ["Reporting", "Completion report to CISO and Audit", "Day 22", "IAM Team"],
                ]
            )},
        ]},
        {"title": "Decision Criteria", "sections": [
            {"title": "Certify Decision Standards", "body": pad(S, "certify decisions", 15)},
            {"title": "Revoke Decision Standards", "body": pad(S, "revoke decisions", 15)},
            {"title": "Escalation Handling", "body": pad(S, "escalation handling", 15)},
        ]},
        {"title": "Automated Controls", "sections": [
            {"title": "IGA Platform Automation", "body": pad(S, "IGA automation", 15)},
            {"title": "AI-Assisted Review", "body": pad(S, "AI-assisted review", 15)},
            {"title": "Auto-Revocation Rules", "body": pad(S, "auto-revocation", 15)},
        ]},
        {"title": "Metrics and KPIs", "sections": [
            {"title": "Certification Completion KPIs", "body": kpi_table(S,
                "Table 5.1 – Access Certification KPIs",
                ["KPI", "Target", "Q1 2026", "Trend"],
                [
                    ["Campaign Completion Rate", "100%", "96.8%", "Improving"],
                    ["On-Time Completion Rate", "95%", "91.2%", "Stable"],
                    ["Revocation Rate", "Report Only", "3.4%", "Stable"],
                    ["Avg Review Time (days)", "<14", "11.2", "Improving"],
                    ["Escalations Triggered", "<5%", "3.8%", "Improving"],
                    ["Auto-Revocations Executed", "N/A", "1,234", "Increasing"],
                ]
            ) + line_chart(S,
                "Figure 5.1 – Quarterly Access Certification Completion Trend (%)",
                ["Q2 2024", "Q3 2024", "Q4 2024", "Q1 2025", "Q2 2025", "Q3 2025", "Q4 2025", "Q1 2026"],
                [
                    ("Completion Rate %", [88, 90, 91, 93, 94, 95, 96, 96.8]),
                    ("On-Time Rate %",    [82, 84, 86, 88, 89, 90, 91, 91.2]),
                ],
            )},
            {"title": "Risk Reduction Metrics", "body": pad(S, "risk reduction metrics", 15)},
            {"title": "Benchmarking", "body": pad(S, "access cert benchmarking", 15)},
        ]},
        {"title": "Audit and Evidence", "sections": [
            {"title": "Audit Evidence Requirements", "body": pad(S, "audit evidence", 15)},
            {"title": "SOX Certification Controls", "body": pad(S, "SOX controls", 15)},
            {"title": "Regulatory Mapping", "body": pad(S, "regulatory mapping", 15)},
        ]},
        {"title": "Exception Management", "sections": [
            {"title": "Exception Handling Process", "body": pad(S, "exception handling", 15)},
            {"title": "Remediation SLAs", "body": kpi_table(S,
                "Table 7.1 – Remediation SLA Matrix",
                ["Finding Severity", "Revocation SLA", "Exception SLA", "Escalation"],
                [
                    ["Critical", "Immediate (2 hrs)", "48 hrs CISO approval", "CTO + Board"],
                    ["High", "24 hours", "5 business days", "CISO"],
                    ["Medium", "72 hours", "10 business days", "Security Manager"],
                    ["Low", "30 days", "30 days", "IAM Manager"],
                ]
            )},
            {"title": "Compensating Controls", "body": pad(S, "compensating controls", 15)},
        ]},
        {"title": "Continuous Improvement", "sections": [
            {"title": "Lessons Learned Process", "body": pad(S, "lessons learned", 15)},
            {"title": "Technology Enhancements", "body": pad(S, "technology improvements", 15)},
            {"title": "Policy Review", "body": pad(S, "policy improvement", 15)},
        ]},
    ]
    return assemble_doc(S, title, subtitle, doc_id, chapters)


def doc_iam_iga(S):
    title = "Identity Governance and Administration (IGA) Framework"
    subtitle = "Enterprise Identity and Access Management"
    doc_id = "IAM-POL-004"
    chapters = [
        {"title": "IGA Program Introduction", "sections": [
            {"title": "IGA Vision and Strategy", "body": pad(S, "IGA strategy", 15)},
            {"title": "IGA Platform Architecture", "body": pad(S, "IGA architecture", 15)},
            {"title": "Identity Data Sources", "body": pad(S, "identity data sources", 15)},
        ]},
        {"title": "Identity Lifecycle Management", "sections": [
            {"title": "Joiner Process", "body": numbered_list(S, [
                "HR system triggers new hire event upon employee record creation.",
                "IGA platform receives identity feed and creates digital identity record.",
                "Role assignment engine applies birthright roles based on job code and department.",
                "Accounts provisioned automatically in connected systems (AD, email, core applications).",
                "Welcome email sent with credentials and security onboarding instructions.",
                "New user enrolled in MFA and security awareness training.",
                "Manager notified of provisioning completion with account summary.",
            ])},
            {"title": "Mover Process", "body": pad(S, "mover process", 15)},
            {"title": "Leaver Process", "body": pad(S, "leaver process", 15)},
        ]},
        {"title": "Role Engineering", "sections": [
            {"title": "Business Role Design", "body": pad(S, "business role design", 15)},
            {"title": "Role Mining and Analytics", "body": pad(S, "role mining", 15)},
            {"title": "Role Catalog Management", "body": pad(S, "role catalog", 15)},
        ]},
        {"title": "Access Request Management", "sections": [
            {"title": "Self-Service Portal", "body": pad(S, "self-service portal", 15)},
            {"title": "Approval Workflow Engine", "body": pad(S, "workflow engine", 15)},
            {"title": "SLA Management", "body": kpi_table(S,
                "Table 4.1 – Access Request SLA Targets",
                ["Request Type", "Target SLA", "Current Avg", "Compliance %"],
                [
                    ["Standard Application Access", "4 hours", "3.2 hours", "94%"],
                    ["Privileged Access", "24 hours", "18 hours", "97%"],
                    ["Contractor Onboarding", "8 hours", "6.5 hours", "91%"],
                    ["Emergency Access", "1 hour", "0.8 hours", "99%"],
                    ["Third-Party Vendor", "48 hours", "36 hours", "88%"],
                ]
            )},
        ]},
        {"title": "Policy Enforcement and Compliance", "sections": [
            {"title": "Policy Engine Configuration", "body": pad(S, "policy engine", 15)},
            {"title": "SoD Rule Management", "body": pad(S, "SoD rules", 15)},
            {"title": "Continuous Compliance Monitoring", "body": pad(S, "continuous compliance", 15)},
        ]},
        {"title": "Reporting and Analytics", "sections": [
            {"title": "IGA Dashboard", "body": pad(S, "IGA dashboard", 15)},
            {"title": "Audit Reports", "body": pad(S, "IGA audit reports", 15)},
            {"title": "Predictive Analytics", "body": pad(S, "predictive analytics", 15)},
        ]},
        {"title": "Third-Party Identity Management", "sections": [
            {"title": "Vendor Identity Lifecycle", "body": pad(S, "vendor identity", 15)},
            {"title": "Third-Party Access Controls", "body": pad(S, "third-party controls", 15)},
            {"title": "Vendor Risk Assessment", "body": pad(S, "vendor risk assessment", 15)},
        ]},
        {"title": "IGA Maturity and Roadmap", "sections": [
            {"title": "Current State Assessment", "body": pad(S, "maturity assessment", 15)},
            {"title": "Target State Architecture", "body": pad(S, "target architecture", 15)},
            {"title": "Implementation Roadmap", "body": kpi_table(S,
                "Table 8.1 – IGA Roadmap Milestones",
                ["Initiative", "Target Quarter", "Status", "Owner"],
                [
                    ["AI-Driven Role Mining", "Q2 2026", "In Planning", "IAM Architect"],
                    ["SCIM 2.0 Integration", "Q3 2026", "In Progress", "Integration Team"],
                    ["Passwordless Authentication", "Q4 2026", "Backlog", "IAM Architect"],
                    ["Continuous Access Evaluation", "Q1 2027", "Backlog", "Security Arch"],
                    ["Decentralized Identity (DID)", "Q3 2027", "Research", "Innovation Team"],
                ]
            )},
        ]},
    ]
    return assemble_doc(S, title, subtitle, doc_id, chapters)


def doc_iam_sso(S):
    title = "Single Sign-On (SSO) and Identity Federation Framework"
    subtitle = "Enterprise Identity and Access Management"
    doc_id = "IAM-POL-005"
    chapters = [
        {"title": "SSO Program Overview", "sections": [
            {"title": "Business Case for SSO", "body": pad(S, "SSO business case", 15)},
            {"title": "Federation Standards", "body": bullet_list(S, [
                "SAML 2.0: XML-based standard for web browser SSO and attribute exchange.",
                "OpenID Connect (OIDC): Identity layer built on OAuth 2.0 for modern applications.",
                "OAuth 2.0: Authorization framework for delegated access to APIs and services.",
                "SCIM 2.0: System for Cross-domain Identity Management for automated provisioning.",
                "WS-Federation: Legacy federation protocol used in Microsoft environments.",
            ])},
            {"title": "Identity Provider Architecture", "body": pad(S, "IdP architecture", 15)},
        ]},
        {"title": "SSO Implementation Standards", "sections": [
            {"title": "Application Integration Requirements", "body": pad(S, "app integration", 15)},
            {"title": "Token Security Standards", "body": pad(S, "token security", 15)},
            {"title": "Session Management", "body": kpi_table(S,
                "Table 2.1 – SSO Session Policy by Risk Level",
                ["Application Risk Level", "Session Timeout", "MFA Required", "Re-auth Trigger"],
                [
                    ["Low (public sites)", "8 hours", "Optional", "Inactivity"],
                    ["Medium (internal apps)", "4 hours", "Required", "Inactivity + Sensitivity"],
                    ["High (financial systems)", "1 hour", "Required + Step-up", "Every transaction"],
                    ["Critical (PAM, security tools)", "30 minutes", "Hardware MFA", "Continuous"],
                ]
            )},
        ]},
        {"title": "Multi-Factor Authentication", "sections": [
            {"title": "MFA Factor Types", "body": pad(S, "MFA factors", 15)},
            {"title": "Adaptive Authentication", "body": pad(S, "adaptive authentication", 15)},
            {"title": "MFA Enrollment and Recovery", "body": pad(S, "MFA enrollment", 15)},
        ]},
        {"title": "External Identity Federation", "sections": [
            {"title": "B2B Federation Design", "body": pad(S, "B2B federation", 15)},
            {"title": "Cloud Provider Integration", "body": pad(S, "cloud IdP integration", 15)},
            {"title": "Partner Onboarding Process", "body": numbered_list(S, [
                "Partner submits federation request through vendor portal.",
                "Security team reviews partner IdP configuration and security posture.",
                "Legal executes Data Processing Agreement (DPA) and federation MOU.",
                "Test federation established in non-production environment.",
                "Security testing validates assertion signing, encryption, and attribute mapping.",
                "Production federation activated with monitoring alerts configured.",
                "Annual review of federation trust relationship and access scope.",
            ])},
        ]},
        {"title": "SSO Security Controls", "sections": [
            {"title": "Assertion Security", "body": pad(S, "assertion security", 15)},
            {"title": "Phishing-Resistant Authentication", "body": pad(S, "phishing resistance", 15)},
            {"title": "Token Theft Prevention", "body": pad(S, "token theft", 15)},
        ]},
        {"title": "Monitoring and Analytics", "sections": [
            {"title": "Authentication Event Logging", "body": pad(S, "auth logging", 15)},
            {"title": "Anomaly Detection", "body": pad(S, "auth anomaly detection", 15)},
            {"title": "SSO KPIs", "body": kpi_table(S,
                "Table 6.1 – SSO Program KPIs",
                ["KPI", "Target", "Current", "Notes"],
                [
                    ["SSO Coverage (%)", "95%", "88.3%", "Legacy apps excluded"],
                    ["MFA Adoption (%)", "100%", "97.6%", "Exemptions pending"],
                    ["Failed Login Rate (%)", "<2%", "1.4%", "Normal range"],
                    ["Avg Login Time (sec)", "<3", "1.8", "Within target"],
                    ["Phishing-Resistant MFA (%)", "80%", "61.2%", "FIDO2 rollout ongoing"],
                ]
            )},
        ]},
        {"title": "Incident Handling for SSO", "sections": [
            {"title": "SSO Outage Response", "body": pad(S, "SSO outage response", 15)},
            {"title": "Account Takeover Response", "body": pad(S, "account takeover", 15)},
            {"title": "Federation Compromise Procedures", "body": pad(S, "federation compromise", 15)},
        ]},
        {"title": "Governance and Compliance", "sections": [
            {"title": "SSO Policy Governance", "body": pad(S, "SSO governance", 15)},
            {"title": "Audit Requirements", "body": pad(S, "SSO audit requirements", 15)},
            {"title": "Roadmap to Zero Trust", "body": pad(S, "zero trust roadmap", 15)},
        ]},
    ]
    return assemble_doc(S, title, subtitle, doc_id, chapters)

# ---------------------------------------------------------------------------
# Fraud Documents
# ---------------------------------------------------------------------------

def _fraud_chapters(S, title, extra=None):
    extra = extra or {}
    domain = title.split()[0]
    return [
        {"title": "Program Overview", "sections": [
            {"title": "Mission and Scope", "body": [
                Paragraph(f"The {title} program establishes a comprehensive approach to detecting, investigating, and preventing fraudulent activities across all business operations and transactional systems. This framework applies to all employees, contractors, and third parties who interact with financial, operational, or data systems.", S["body"]),
                Paragraph("The Association of Certified Fraud Examiners (ACFE) estimates that organizations lose approximately 5% of annual revenue to fraud, with insider threats accounting for a disproportionate share of losses due to the perpetrator's knowledge of internal controls and detection gaps.", S["body"]),
            ] + pad(S, f"{domain} fraud program", 15)},
            {"title": "Regulatory Context", "body": pad(S, "fraud regulation", 15)},
            {"title": "Organizational Roles", "body": bullet_list(S, [
                "Chief Compliance Officer (CCO): Program ownership and regulatory interface.",
                "Fraud Investigations Unit: Conducts formal fraud investigations.",
                "Security Operations Center (SOC): Real-time monitoring and alert triage.",
                "Internal Audit: Independent validation of fraud control effectiveness.",
                "HR and Legal: Support disciplinary and legal proceedings.",
                "Business Unit Managers: First-line control owners and anomaly reporters.",
            ])},
        ]},
        {"title": "Detection Methodology", "sections": [
            {"title": "Detection Layers", "body": pad(S, "fraud detection layers", 15)},
            {"title": "Rule-Based Detection", "body": pad(S, "rule-based detection", 15)},
            {"title": "Machine Learning Models", "body": pad(S, "ML fraud models", 15)},
        ]},
        {"title": "Risk Indicators and Typologies", "sections": [
            {"title": "Red Flag Indicators", "body": bullet_list(S, [
                "Unusual transaction volumes or values outside historical norms.",
                "Transactions occurring outside normal business hours without justification.",
                "Repeated small transactions below approval thresholds (structuring).",
                "Access to systems or data inconsistent with job function.",
                "Override of system controls or audit trail manipulation.",
                "Expense claims with missing or altered receipts.",
                "Conflicts of interest in vendor approval processes.",
                "Sudden lifestyle changes inconsistent with compensation level.",
            ])},
            {"title": "Fraud Typology Catalog", "body": pad(S, "fraud typologies", 15)},
            {"title": "Industry Benchmarks", "body": kpi_table(S,
                "Table 3.1 – Fraud Loss Benchmarks by Category",
                ["Fraud Type", "Avg Loss per Incident", "Median Duration", "Detection Method"],
                [
                    ["Asset Misappropriation", "$100,000", "18 months", "Tips, audit"],
                    ["Financial Statement Fraud", "$954,000", "24 months", "Analytics"],
                    ["Corruption/Bribery", "$200,000", "18 months", "Tips"],
                    ["Payroll Fraud", "$90,000", "30 months", "Review"],
                    ["Expense Reimbursement", "$26,000", "24 months", "Analytics, tips"],
                    ["Billing Schemes", "$100,000", "24 months", "Audit"],
                ]
            ) + extra.get((2, 2), [])},
        ]},
        {"title": "Monitoring and Analytics Platform", "sections": [
            {"title": "Data Sources and Integration", "body": pad(S, "fraud data sources", 15)},
            {"title": "Real-Time Monitoring Rules", "body": pad(S, "monitoring rules", 15)},
            {"title": "Analytics Dashboard", "body": pad(S, "fraud analytics dashboard", 15)},
        ]},
        {"title": "Alert Management", "sections": [
            {"title": "Alert Classification", "body": kpi_table(S,
                "Table 5.1 – Fraud Alert Severity Matrix",
                ["Severity", "Criteria", "Response SLA", "Escalation"],
                [
                    ["Critical", "Active fraud, loss >$50K", "15 minutes", "CCO + Legal"],
                    ["High", "Strong indicators, potential loss >$10K", "1 hour", "Fraud Manager"],
                    ["Medium", "Suspicious patterns, investigation needed", "4 hours", "Analyst"],
                    ["Low", "Weak signals, review recommended", "24 hours", "Analyst"],
                    ["Informational", "Policy anomaly, no immediate risk", "72 hours", "Business Unit"],
                ]
            )},
            {"title": "Triage Procedures", "body": pad(S, "alert triage procedures", 15)},
            {"title": "False Positive Management", "body": pad(S, "false positive management", 15)},
        ]},
        {"title": "Investigation Procedures", "sections": [
            {"title": "Investigation Initiation", "body": numbered_list(S, [
                "Fraud alert or tip received and logged in case management system.",
                "Preliminary triage performed to assess credibility and urgency.",
                "Case formally opened with unique ID and assigned investigator.",
                "Evidence preservation order issued to prevent data destruction.",
                "Legal hold placed on relevant electronic communications and records.",
                "Investigation plan developed with scope, objectives, and timeline.",
                "Stakeholder notifications made per established protocols.",
            ])},
            {"title": "Digital Forensics", "body": pad(S, "digital forensics", 15)},
            {"title": "Interview Techniques", "body": pad(S, "investigation interviews", 15)},
        ]},
        {"title": "KPIs and Performance Metrics", "sections": [
            {"title": "Fraud Detection KPIs", "body": kpi_table(S,
                "Table 7.1 – Fraud Program KPI Dashboard",
                ["KPI", "Target", "Current", "Trend"],
                [
                    ["Fraud Loss Rate (% revenue)", "<0.1%", "0.08%", "Improving"],
                    ["Mean Time to Detect (days)", "<30", "22", "Improving"],
                    ["Mean Time to Contain (days)", "<5", "3.2", "Stable"],
                    ["Alert-to-Case Conversion Rate", "15-25%", "19%", "Stable"],
                    ["Investigation Close Rate (30d)", ">80%", "84%", "Improving"],
                    ["False Positive Rate", "<60%", "54%", "Improving"],
                ]
            ) + extra.get((6, 0), [])},
            {"title": "Operational Efficiency", "body": pad(S, "operational efficiency", 15)},
            {"title": "Executive Reporting", "body": pad(S, "executive fraud reporting", 15)},
        ]},
        {"title": "Governance and Continuous Improvement", "sections": [
            {"title": "Fraud Risk Governance", "body": pad(S, "fraud governance", 15)},
            {"title": "Control Effectiveness Testing", "body": pad(S, "control effectiveness testing", 15)},
            {"title": "Program Maturity Roadmap", "body": pad(S, "fraud program roadmap", 15)},
        ]},
    ]


def doc_fraud(S, title, subtitle, doc_id, extra=None):
    return assemble_doc(S, title, subtitle, doc_id, _fraud_chapters(S, title, extra))

# ---------------------------------------------------------------------------
# Vulnerability Management Documents
# ---------------------------------------------------------------------------

def _vuln_chapters(S, title, extra=None):
    extra = extra or {}
    return [
        {"title": "Vulnerability Management Program Overview", "sections": [
            {"title": "Program Mission and Objectives", "body": [
                Paragraph(f"The {title} framework establishes the enterprise approach to identifying, assessing, prioritizing, and remediating security vulnerabilities across the entire technology estate. Effective vulnerability management reduces the organization's exposure to exploitation by shrinking the attack surface available to adversaries.", S["body"]),
                Paragraph("This program adheres to NIST SP 800-40, CIS Controls v8 Control 7 (Continuous Vulnerability Management), and CVSS v3.1 for vulnerability rating.", S["body"]),
            ] + pad(S, "vulnerability management", 15)},
            {"title": "Scope and Coverage", "body": pad(S, "VM scope", 15)},
            {"title": "Roles and Responsibilities", "body": bullet_list(S, [
                "Vulnerability Management Team: Program ownership, scanning, and reporting.",
                "Asset Owners: Accountability for remediation within their asset scope.",
                "Patch Management Team: Deployment of patches and configuration changes.",
                "SOC: Threat intelligence integration and exploit monitoring.",
                "CISO: Program oversight and exception approval.",
                "IT Operations: Infrastructure and server patch deployment.",
                "Application Security: Code-level vulnerability remediation.",
            ])},
        ]},
        {"title": "Vulnerability Discovery and Assessment", "sections": [
            {"title": "Scanning Architecture", "body": pad(S, "scanning architecture", 15)},
            {"title": "Authenticated vs Unauthenticated Scanning", "body": pad(S, "scan types", 15)},
            {"title": "Scan Coverage Requirements", "body": kpi_table(S,
                "Table 2.1 – Scanning Coverage Requirements",
                ["Asset Type", "Scan Frequency", "Scan Type", "Coverage Target"],
                [
                    ["Internet-Facing Systems", "Daily", "Authenticated + External", "100%"],
                    ["Internal Servers", "Weekly", "Authenticated", "100%"],
                    ["Workstations/Endpoints", "Weekly", "Agent-Based", "98%"],
                    ["Network Devices", "Weekly", "SNMP + Authenticated", "95%"],
                    ["Cloud Workloads", "Continuous (CSPM)", "Agent + API", "100%"],
                    ["OT/ICS Systems", "Monthly", "Passive", "90%"],
                ]
            )},
        ]},
        {"title": "Vulnerability Scoring and Prioritization", "sections": [
            {"title": "CVSS Framework", "body": [
                Paragraph("The Common Vulnerability Scoring System (CVSS) v3.1 provides a standardized method for rating vulnerability severity. CVSS scores range from 0.0 to 10.0: None (0.0), Low (0.1-3.9), Medium (4.0-6.9), High (7.0-8.9), and Critical (9.0-10.0).", S["body"]),
                Paragraph("The base score reflects intrinsic characteristics, while temporal and environmental scores allow contextual adjustment for asset criticality, exploitability intelligence, and compensating controls.", S["body"]),
            ] + pad(S, "CVSS scoring", 15) + extra.get((2, 0), [])},
            {"title": "Risk-Based Prioritization", "body": pad(S, "risk prioritization", 15)},
            {"title": "Threat Intelligence Integration", "body": pad(S, "threat intelligence", 15)},
        ]},
        {"title": "Remediation Framework", "sections": [
            {"title": "Remediation SLA Policy", "body": kpi_table(S,
                "Table 4.1 – Vulnerability Remediation SLA Matrix",
                ["CVSS Severity", "Internet-Facing SLA", "Internal SLA", "Escalation"],
                [
                    ["Critical (9.0-10.0)", "24 hours", "72 hours", "CISO immediate"],
                    ["High (7.0-8.9)", "7 days", "14 days", "VP Engineering"],
                    ["Medium (4.0-6.9)", "30 days", "60 days", "Security Manager"],
                    ["Low (0.1-3.9)", "90 days", "180 days", "Asset Owner"],
                ]
            )},
            {"title": "Remediation Options", "body": bullet_list(S, [
                "Patch/Update: Apply vendor-supplied security patch or software update.",
                "Configuration Change: Harden system configuration to eliminate vulnerability.",
                "Workaround: Implement temporary mitigation while patch is tested.",
                "Network Isolation: Restrict network access to limit exploitability.",
                "Accept Risk: Formally accept risk with documented justification and CISO approval.",
                "Decommission: Remove asset from service if remediation is not feasible.",
            ])},
            {"title": "Patch Testing and Deployment", "body": pad(S, "patch deployment", 15)},
        ]},
        {"title": "Metrics and Reporting", "sections": [
            {"title": "VM KPI Dashboard", "body": kpi_table(S,
                "Table 5.1 – Vulnerability Management KPIs",
                ["KPI", "Target", "Current", "Trend"],
                [
                    ["Critical Vuln Remediation Rate (SLA)", "100%", "97.2%", "Improving"],
                    ["High Vuln Remediation Rate (SLA)", "95%", "91.4%", "Stable"],
                    ["Mean Time to Remediate Critical (hrs)", "<24", "31", "Needs Attention"],
                    ["Scan Coverage (%)", "98%", "95.3%", "Improving"],
                    ["Vuln Recurrence Rate", "<5%", "3.1%", "Improving"],
                    ["SLA Exception Rate", "<2%", "1.8%", "Stable"],
                ]
            ) + extra.get((4, 0), [])},
            {"title": "Trend Analysis", "body": pad(S, "vulnerability trend analysis", 15) + extra.get((4, 1), [])},
            {"title": "Executive Risk Reporting", "body": pad(S, "VM executive reporting", 15)},
        ]},
        {"title": "Tooling and Automation", "sections": [
            {"title": "Vulnerability Scanner Configuration", "body": pad(S, "scanner configuration", 15)},
            {"title": "ITSM Integration", "body": pad(S, "ITSM integration", 15)},
            {"title": "Automation Workflows", "body": pad(S, "VM automation workflows", 15)},
        ]},
        {"title": "Exception and Risk Acceptance", "sections": [
            {"title": "Exception Process", "body": numbered_list(S, [
                "Asset owner submits Exception Request via GRC portal with business justification.",
                "Vulnerability Management team validates CVSS score and exploitability.",
                "Risk Management evaluates compensating controls and residual risk level.",
                "CISO approval required for Critical and High severity exceptions.",
                "Approved exceptions documented in risk register with expiry date.",
                "Exceptions reviewed monthly by Vulnerability Management team.",
                "Exceptions auto-expire after 90 days and must be re-submitted.",
            ])},
            {"title": "Compensating Controls Standards", "body": pad(S, "compensating controls", 15)},
            {"title": "Risk Acceptance Authority Matrix", "body": pad(S, "risk acceptance authority", 15)},
        ]},
        {"title": "Governance and Compliance", "sections": [
            {"title": "VM Governance Structure", "body": pad(S, "VM governance", 15)},
            {"title": "Compliance Mapping", "body": kpi_table(S,
                "Table 8.1 – Regulatory Mapping for Vulnerability Management",
                ["Framework", "Control Reference", "VM Requirement", "Evidence"],
                [
                    ["PCI DSS", "Req 6.3", "Critical scan coverage", "Scan reports"],
                    ["HIPAA", "§164.308(a)(1)", "Risk analysis", "VM risk register"],
                    ["NIST CSF", "ID.RA-1", "Asset vulnerability identification", "Scan data"],
                    ["ISO 27001", "A.12.6.1", "Technical vulnerability management", "Policy doc"],
                    ["CIS Controls", "Control 7", "Continuous VM program", "KPI reports"],
                ]
            )},
            {"title": "Program Review and Improvement", "body": pad(S, "VM program review", 15)},
        ]},
    ]


def doc_vuln(S, title, subtitle, doc_id, extra=None):
    return assemble_doc(S, title, subtitle, doc_id, _vuln_chapters(S, title, extra))

# ---------------------------------------------------------------------------
# UEBA Documents
# ---------------------------------------------------------------------------

def _ueba_chapters(S, title, extra=None):
    extra = extra or {}
    return [
        {"title": "UEBA Program Introduction", "sections": [
            {"title": "UEBA Vision and Objectives", "body": [
                Paragraph(f"The {title} framework defines the enterprise approach to leveraging User and Entity Behavior Analytics (UEBA) as a core security operations capability. UEBA applies machine learning, statistical modeling, and advanced analytics to detect anomalies in user behavior, device activity, and entity interactions.", S["body"]),
                Paragraph("Unlike signature-based detection that relies on known threat patterns, UEBA establishes dynamic behavioral baselines for each user and entity, enabling detection of novel threats that evade traditional controls.", S["body"]),
            ] + pad(S, "UEBA program", 15)},
            {"title": "UEBA Data Sources", "body": bullet_list(S, [
                "Active Directory and LDAP: Authentication events, group membership changes.",
                "Network Flow Data (NetFlow/IPFIX): Connection patterns, data volumes, destinations.",
                "Endpoint Detection and Response (EDR): Process execution, file access, registry changes.",
                "Email Security Gateway: Communication patterns, attachment analysis.",
                "DLP Platform: Data access and exfiltration attempts.",
                "HR System: Employment status, role changes, termination events.",
                "Badge/Physical Access: Building entry/exit patterns.",
                "Cloud Application Logs: SaaS activity from O365, Salesforce, etc.",
                "VPN Logs: Remote access patterns and geographic anomalies.",
                "PAM Logs: Privileged session activity.",
            ])},
            {"title": "ML Model Architecture", "body": pad(S, "UEBA ML models", 15)},
        ]},
        {"title": "Baseline Profiling", "sections": [
            {"title": "Peer Group Analysis", "body": pad(S, "peer group analysis", 15)},
            {"title": "Individual Baseline Construction", "body": pad(S, "individual baselines", 15)},
            {"title": "Baseline Tuning and Maintenance", "body": kpi_table(S,
                "Table 2.1 – Baseline Profile Parameters",
                ["Behavior Category", "Baseline Window", "Update Frequency", "Sensitivity"],
                [
                    ["Login Patterns", "90 days", "Daily", "Medium"],
                    ["Data Access Volume", "30 days", "Hourly", "High"],
                    ["Network Connections", "30 days", "Daily", "Medium"],
                    ["Application Usage", "60 days", "Weekly", "Low"],
                    ["Email Communication", "90 days", "Daily", "Medium"],
                    ["File Operations", "30 days", "Real-time", "High"],
                ]
            )},
        ]},
        {"title": "Anomaly Detection Models", "sections": [
            {"title": "Statistical Anomaly Detection", "body": pad(S, "statistical anomaly detection", 15) + extra.get((2, 0), [])},
            {"title": "Machine Learning Classifiers", "body": pad(S, "ML classifiers", 15)},
            {"title": "Rule Augmentation", "body": pad(S, "rule augmentation", 15)},
        ]},
        {"title": "Risk Scoring Framework", "sections": [
            {"title": "Entity Risk Score Calculation", "body": [
                Paragraph("Each user and entity receives a dynamic risk score on a 0-100 scale, calculated through a weighted combination of anomaly signals, threat intelligence matches, and contextual factors. The risk score decays over time in the absence of new anomalous activity.", S["body"]),
            ] + pad(S, "risk score calculation", 15) + extra.get((3, 0), [])},
            {"title": "Risk Score Thresholds", "body": kpi_table(S,
                "Table 4.1 – Risk Score Action Thresholds",
                ["Score Range", "Risk Level", "Action", "Review SLA"],
                [
                    ["0-39", "Low", "Monitor only", "No action required"],
                    ["40-59", "Medium", "Watchlist", "Weekly analyst review"],
                    ["60-79", "High", "Alert generated", "24-hour investigation"],
                    ["80-89", "Critical", "Automatic case opened", "4-hour investigation"],
                    ["90-100", "Imminent Threat", "Account suspension + SOC page", "Immediate"],
                ]
            )},
            {"title": "Score Aggregation and Correlation", "body": pad(S, "score aggregation", 15)},
        ]},
        {"title": "Alert Triage and Investigation", "sections": [
            {"title": "Alert Queue Management", "body": pad(S, "alert queue management", 15)},
            {"title": "Investigation Playbooks", "body": numbered_list(S, [
                "Retrieve full behavioral timeline for the entity from UEBA platform.",
                "Review recent access patterns against peer group baseline.",
                "Check HR system for any pending transfers, PIPs, or termination notices.",
                "Review badge access logs for unusual building entry patterns.",
                "Examine email and communication metadata for external contact anomalies.",
                "Query SIEM for correlated events from other security tools.",
                "Interview manager if risk score exceeds 75 without clear technical explanation.",
                "Document findings and risk determination in case management system.",
            ])},
            {"title": "Escalation Procedures", "body": pad(S, "UEBA escalation procedures", 15)},
        ]},
        {"title": "Insider Threat Detection", "sections": [
            {"title": "Insider Threat Taxonomy", "body": pad(S, "insider threat taxonomy", 15)},
            {"title": "Pre-Departure Indicator Detection", "body": pad(S, "pre-departure indicators", 15)},
            {"title": "Case Studies and Scenarios", "body": pad(S, "UEBA scenarios", 15)},
        ]},
        {"title": "UEBA KPIs and Performance", "sections": [
            {"title": "Detection Performance Metrics", "body": kpi_table(S,
                "Table 7.1 – UEBA Detection Performance KPIs",
                ["KPI", "Target", "Current", "Notes"],
                [
                    ["True Positive Rate", ">40%", "47%", "Above target"],
                    ["False Positive Rate", "<60%", "53%", "Improving"],
                    ["Mean Time to Detect (days)", "<5", "3.8", "On target"],
                    ["Alert Volume (daily)", "<100", "76", "Within range"],
                    ["High-Risk Entity Reviews (monthly)", "N/A", "234", "Tracking"],
                    ["Cases Converted to IR", "N/A", "12", "Monthly avg"],
                ]
            ) + extra.get((6, 0), [])},
            {"title": "Model Accuracy Monitoring", "body": pad(S, "model accuracy monitoring", 15)},
            {"title": "Continuous Improvement", "body": pad(S, "UEBA improvement", 15)},
        ]},
        {"title": "Governance and Privacy", "sections": [
            {"title": "Employee Privacy Considerations", "body": [
                Paragraph("UEBA monitoring of employee behavior raises important privacy considerations that must be balanced against legitimate security interests. The organization conducts monitoring in accordance with GDPR, CCPA, and local employment regulations. Monitoring is limited to work-related systems and company-owned assets.", S["body"]),
                Paragraph("Employees are informed of monitoring through the Acceptable Use Policy acknowledged at onboarding and annually thereafter. UEBA data is accessible only to authorized security personnel with a documented need-to-know.", S["body"]),
            ] + pad(S, "privacy considerations", 15)},
            {"title": "Data Retention for UEBA", "body": pad(S, "UEBA data retention", 15)},
            {"title": "Legal and HR Coordination", "body": pad(S, "legal HR coordination", 15)},
        ]},
    ]


def doc_ueba(S, title, subtitle, doc_id, extra=None):
    return assemble_doc(S, title, subtitle, doc_id, _ueba_chapters(S, title, extra))

# ---------------------------------------------------------------------------
# ASM Documents
# ---------------------------------------------------------------------------

def _asm_chapters(S, title, extra=None):
    extra = extra or {}
    return [
        {"title": "ASM Program Introduction", "sections": [
            {"title": "Attack Surface Management Overview", "body": [
                Paragraph(f"The {title} program establishes the enterprise capability to continuously discover, inventory, assess, and manage the organization's digital attack surface. This encompasses all internet-facing assets, cloud resources, third-party connections, and shadow IT that could be exploited by adversaries.", S["body"]),
                Paragraph("Modern enterprises face a rapidly expanding attack surface driven by cloud adoption, remote work, M&A activity, and shadow IT. ASM provides the continuous visibility required to identify exposures before adversaries exploit them.", S["body"]),
            ] + pad(S, "ASM program", 15)},
            {"title": "ASM Scope and Asset Classes", "body": bullet_list(S, [
                "Internet-Facing Infrastructure: Web servers, APIs, VPNs, remote access portals.",
                "Cloud Assets: EC2 instances, S3 buckets, Azure resources, GCP services.",
                "Domains and Subdomains: All registered domains and DNS records.",
                "SSL/TLS Certificates: Certificate inventory and expiry monitoring.",
                "Third-Party and Supply Chain: Vendor-hosted systems processing company data.",
                "Shadow IT: Unauthorized cloud services and applications.",
                "Acquired Entities: Assets from mergers and acquisitions not yet integrated.",
            ])},
            {"title": "ASM Tool Architecture", "body": pad(S, "ASM tool architecture", 15)},
        ]},
        {"title": "Asset Discovery and Inventory", "sections": [
            {"title": "Continuous Discovery Methods", "body": pad(S, "asset discovery methods", 15)},
            {"title": "Discovery Data Sources", "body": pad(S, "discovery data sources", 15)},
            {"title": "Asset Classification", "body": kpi_table(S,
                "Table 2.1 – Asset Inventory by Category",
                ["Asset Category", "Discovery Method", "Scan Frequency", "Ownership"],
                [
                    ["Web Applications", "DAST + Spider", "Daily", "App Security Team"],
                    ["Cloud Resources", "CSPM + API", "Continuous", "Cloud Security Team"],
                    ["Network Infrastructure", "Port Scan + BGP", "Weekly", "Network Team"],
                    ["Certificates", "CT Log + Scan", "Daily", "PKI Team"],
                    ["Domains/DNS", "OSINT + Registrar", "Daily", "IT Operations"],
                    ["Third-Party Services", "Questionnaire + Scan", "Monthly", "Vendor Mgmt"],
                ]
            )},
        ]},
        {"title": "Exposure Assessment", "sections": [
            {"title": "Exposure Scoring Methodology", "body": pad(S, "exposure scoring methodology", 15) + extra.get((2, 0), [])},
            {"title": "Critical Exposure Types", "body": bullet_list(S, [
                "Open ports and services exposing internal infrastructure.",
                "Misconfigured S3 buckets or storage containers with public access.",
                "Expired or weak SSL/TLS certificates.",
                "Vulnerable software versions with known CVEs.",
                "Default credentials on internet-facing devices.",
                "Exposed administrative interfaces (admin panels, RDP, SSH).",
                "Leaked credentials or API keys in public repositories.",
                "Orphaned DNS records pointing to decommissioned resources.",
            ])},
            {"title": "Exposure Prioritization", "body": pad(S, "exposure prioritization", 15)},
        ]},
        {"title": "Cloud Attack Surface", "sections": [
            {"title": "Cloud Misconfiguration Detection", "body": pad(S, "cloud misconfiguration", 15)},
            {"title": "Shadow IT Discovery", "body": pad(S, "shadow IT discovery", 15)},
            {"title": "Cloud Security Posture KPIs", "body": kpi_table(S,
                "Table 4.1 – Cloud ASM KPIs",
                ["KPI", "Target", "Current", "Trend"],
                [
                    ["Public Cloud Assets Inventoried (%)", "100%", "96.4%", "Improving"],
                    ["Critical Misconfigurations Open", "<10", "7", "Improving"],
                    ["Unauthorized Cloud Services", "<20", "34", "Needs Attention"],
                    ["Avg Exposure Closure Time (days)", "<7", "9.2", "Improving"],
                    ["Certificate Expiry Alerts", "0", "2", "Action Required"],
                ]
            ) + extra.get((3, 2), [])},
        ]},
        {"title": "Third-Party Attack Surface", "sections": [
            {"title": "Vendor Risk Assessment", "body": pad(S, "vendor risk assessment", 15)},
            {"title": "Supply Chain Exposure Monitoring", "body": pad(S, "supply chain monitoring", 15)},
            {"title": "Fourth-Party Risk", "body": pad(S, "fourth-party risk", 15)},
        ]},
        {"title": "Remediation Management", "sections": [
            {"title": "Remediation Workflow", "body": numbered_list(S, [
                "ASM platform identifies and scores new exposure.",
                "Automated ticket created in ITSM with asset owner assigned.",
                "Asset owner acknowledges finding within defined SLA.",
                "Remediation plan submitted with target completion date.",
                "ASM platform performs validation scan upon owner notification of fix.",
                "Finding closed upon confirmed remediation; tracking metrics updated.",
                "Recurrence trigger activated if same exposure re-emerges within 30 days.",
            ])},
            {"title": "SLA Framework", "body": pad(S, "ASM SLA framework", 15)},
            {"title": "Exception Management", "body": pad(S, "ASM exception management", 15)},
        ]},
        {"title": "ASM KPIs and Reporting", "sections": [
            {"title": "ASM Performance Dashboard", "body": kpi_table(S,
                "Table 7.1 – ASM Program KPIs",
                ["KPI", "Target", "Current", "Owner"],
                [
                    ["Attack Surface Coverage (%)", "100%", "94.7%", "ASM Team"],
                    ["Critical Exposures (SLA met)", "100%", "93.2%", "Asset Owners"],
                    ["Mean Time to Discover (hrs)", "<24", "18", "ASM Team"],
                    ["Mean Time to Remediate (days)", "<14", "11.3", "IT Operations"],
                    ["Shadow IT Assets Resolved", "90%", "67%", "IT Governance"],
                    ["New Asset Discovery Rate", "N/A", "47/week", "Tracking"],
                ]
            )},
            {"title": "Trend and Risk Reporting", "body": pad(S, "ASM trend reporting", 15)},
            {"title": "Executive Briefing Content", "body": pad(S, "ASM executive briefing", 15)},
        ]},
        {"title": "Governance and Program Maturity", "sections": [
            {"title": "ASM Governance Model", "body": pad(S, "ASM governance model", 15)},
            {"title": "Maturity Assessment", "body": kpi_table(S,
                "Table 8.1 – ASM Maturity Model",
                ["Level", "Capability", "Current Status", "Target"],
                [
                    ["1 - Reactive", "Manual asset inventory", "Completed", "N/A"],
                    ["2 - Defined", "Periodic scanning", "Completed", "N/A"],
                    ["3 - Managed", "Continuous discovery", "Current", "N/A"],
                    ["4 - Proactive", "Risk-based prioritization", "In Progress", "Q3 2026"],
                    ["5 - Optimized", "Predictive exposure analytics", "Planned", "Q1 2027"],
                ]
            )},
            {"title": "Technology Roadmap", "body": pad(S, "ASM technology roadmap", 15)},
        ]},
    ]


def doc_asm(S, title, subtitle, doc_id, extra=None):
    return assemble_doc(S, title, subtitle, doc_id, _asm_chapters(S, title, extra))

# ---------------------------------------------------------------------------
# SIEM Documents
# ---------------------------------------------------------------------------

def _siem_chapters(S, title, extra=None):
    extra = extra or {}
    return [
        {"title": "SIEM Program Overview", "sections": [
            {"title": "SIEM Architecture and Mission", "body": [
                Paragraph(f"The {title} framework defines the enterprise approach to Security Information and Event Management (SIEM), the foundational platform for security monitoring, threat detection, and incident response. The SIEM aggregates security telemetry from across the technology estate, normalizes event data, and applies detection logic to identify threats in real time.", S["body"]),
                Paragraph("The enterprise SIEM processes approximately 500,000 events per second (EPS) from over 2,000 log sources spanning endpoint, network, cloud, application, and identity domains.", S["body"]),
            ] + pad(S, "SIEM program", 15)},
            {"title": "Log Source Coverage", "body": bullet_list(S, [
                "Endpoint Security: EDR, antivirus, DLP agent logs.",
                "Network Security: Firewall, IDS/IPS, proxy, DNS, NetFlow.",
                "Identity Systems: Active Directory, LDAP, IAM, PAM session logs.",
                "Cloud Platforms: AWS CloudTrail, Azure Activity Logs, GCP Audit Logs.",
                "Applications: Web application firewall, authentication logs, API gateway.",
                "Email Security: Anti-phishing gateway, mail flow logs.",
                "Physical Security: Badge reader events correlated with logical access.",
            ])},
            {"title": "SIEM Team Structure", "body": pad(S, "SIEM team structure", 15)},
        ]},
        {"title": "Log Management and Normalization", "sections": [
            {"title": "Log Collection Architecture", "body": pad(S, "log collection architecture", 15)},
            {"title": "Normalization and Parsing", "body": pad(S, "log normalization and parsing", 15)},
            {"title": "Log Retention Policy", "body": kpi_table(S,
                "Table 2.1 – Log Retention Requirements",
                ["Log Type", "Hot Storage", "Cold Storage", "Total Retention"],
                [
                    ["Security Events (SIEM)", "90 days", "7 years", "7 years"],
                    ["Authentication Logs", "90 days", "7 years", "7 years"],
                    ["Network Flow Data", "30 days", "1 year", "1 year"],
                    ["Endpoint Telemetry", "30 days", "1 year", "1 year"],
                    ["Cloud Audit Logs", "90 days", "3 years", "3 years"],
                    ["Application Logs", "30 days", "1 year", "1 year"],
                ]
            )},
        ]},
        {"title": "Detection Engineering", "sections": [
            {"title": "Use Case Development Process", "body": numbered_list(S, [
                "Threat intelligence team identifies new threat scenario or TTPs.",
                "Detection engineer maps scenario to MITRE ATT&CK framework.",
                "Data source requirements identified and availability confirmed.",
                "Detection logic developed in SIEM query language.",
                "Unit testing performed against historical data to validate detection.",
                "False positive analysis conducted with tuning applied.",
                "Use case deployed to production with alert routing configured.",
                "Performance metrics tracked for 30 days post-deployment.",
                "Annual review of use case relevance and effectiveness.",
            ])},
            {"title": "MITRE ATT&CK Coverage", "body": pad(S, "MITRE ATT&CK coverage", 15) + extra.get((2, 1), [])},
            {"title": "Detection Rule Library", "body": pad(S, "detection rule library", 15)},
        ]},
        {"title": "Alert Management", "sections": [
            {"title": "Alert Severity Classification", "body": kpi_table(S,
                "Table 4.1 – SIEM Alert Severity Matrix",
                ["Priority", "CVSS / Score", "Response SLA", "Analyst Action"],
                [
                    ["P1 - Critical", "9.0+", "15 minutes", "Immediate IR escalation"],
                    ["P2 - High", "7.0-8.9", "1 hour", "Full investigation"],
                    ["P3 - Medium", "4.0-6.9", "4 hours", "Triage and assess"],
                    ["P4 - Low", "0.1-3.9", "24 hours", "Log and review"],
                    ["P5 - Informational", "0.0", "72 hours", "Trending analysis"],
                ]
            ) + extra.get((3, 0), [])},
            {"title": "Alert Triage Workflow", "body": pad(S, "alert triage workflow", 15)},
            {"title": "SOAR Integration", "body": pad(S, "SOAR integration", 15)},
        ]},
        {"title": "Threat Correlation", "sections": [
            {"title": "Correlation Rule Design", "body": pad(S, "correlation rule design", 15)},
            {"title": "Multi-Stage Attack Detection", "body": pad(S, "multi-stage attack detection", 15)},
            {"title": "Kill Chain Mapping", "body": pad(S, "kill chain mapping", 15)},
        ]},
        {"title": "SIEM Performance Tuning", "sections": [
            {"title": "False Positive Reduction", "body": pad(S, "false positive reduction", 15)},
            {"title": "Whitelist Management", "body": pad(S, "whitelist management", 15)},
            {"title": "SIEM KPIs", "body": kpi_table(S,
                "Table 6.1 – SIEM Operational KPIs",
                ["KPI", "Target", "Current", "Trend"],
                [
                    ["Events Per Second (EPS)", "500K", "487K", "Stable"],
                    ["Alert Volume (daily)", "<500", "423", "Within target"],
                    ["False Positive Rate", "<50%", "44%", "Improving"],
                    ["Mean Time to Detect (min)", "<60", "34", "Improving"],
                    ["Use Case Coverage (MITRE)", ">70%", "68%", "Improving"],
                    ["Log Source Uptime (%)", "99.9%", "99.7%", "Stable"],
                ]
            ) + extra.get((5, 2), [])},
        ]},
        {"title": "Incident Response Integration", "sections": [
            {"title": "IR Escalation from SIEM", "body": pad(S, "IR escalation from SIEM", 15)},
            {"title": "Forensic Evidence Preservation", "body": pad(S, "forensic evidence preservation", 15)},
            {"title": "Post-Incident Use Case Updates", "body": pad(S, "post-incident use case updates", 15)},
        ]},
        {"title": "Governance and Compliance", "sections": [
            {"title": "SIEM Governance Model", "body": pad(S, "SIEM governance model", 15)},
            {"title": "Compliance Reporting", "body": pad(S, "SIEM compliance reporting", 15)},
            {"title": "Continuous Improvement", "body": pad(S, "SIEM continuous improvement", 15)},
        ]},
    ]


def doc_siem(S, title, subtitle, doc_id, extra=None):
    return assemble_doc(S, title, subtitle, doc_id, _siem_chapters(S, title, extra))

# ---------------------------------------------------------------------------
# Data Protection Documents
# ---------------------------------------------------------------------------

def _dp_chapters(S, title, extra=None):
    extra = extra or {}
    return [
        {"title": "Data Protection Program Overview", "sections": [
            {"title": "Program Mission", "body": [
                Paragraph(f"The {title} framework establishes the enterprise approach to protecting sensitive data throughout its lifecycle—from creation and storage through processing, transmission, and disposal. Data protection is a fundamental obligation driven by regulatory requirements, contractual commitments, and the organization's duty of care.", S["body"]),
                Paragraph("This program aligns with GDPR, CCPA, HIPAA, PCI DSS, ISO/IEC 27001:2022 Annex A.8, and NIST SP 800-53 SC control family.", S["body"]),
            ] + pad(S, "data protection program", 15)},
            {"title": "Data Protection Principles", "body": bullet_list(S, [
                "Lawfulness, Fairness, and Transparency: Data processed only on valid legal bases.",
                "Purpose Limitation: Data collected for specified, explicit, and legitimate purposes only.",
                "Data Minimization: Only data necessary for the stated purpose is collected.",
                "Accuracy: Data must be kept accurate and up to date.",
                "Storage Limitation: Data retained only as long as necessary.",
                "Integrity and Confidentiality: Appropriate technical and organizational controls applied.",
                "Accountability: Demonstrable compliance with data protection principles.",
            ])},
            {"title": "Regulatory Landscape", "body": pad(S, "data protection regulations", 15)},
        ]},
        {"title": "Data Classification", "sections": [
            {"title": "Classification Framework", "body": kpi_table(S,
                "Table 2.1 – Data Classification Tiers",
                ["Classification", "Description", "Examples", "Handling Requirement"],
                [
                    ["Public", "Approved for public disclosure", "Marketing, press releases", "Standard controls"],
                    ["Internal", "Business use only", "Internal memos, procedures", "Access controls"],
                    ["Confidential", "Sensitive business data", "Financial data, strategies", "Encryption + ACL"],
                    ["Restricted", "Highest sensitivity", "PII, PHI, PCI data, IP", "Full DLP + encryption"],
                ]
            ) + extra.get((1, 0), [])},
            {"title": "Data Discovery and Inventory", "body": pad(S, "data discovery and inventory", 15)},
            {"title": "Classification Tooling", "body": pad(S, "classification tooling", 15)},
        ]},
        {"title": "Data Loss Prevention", "sections": [
            {"title": "DLP Program Architecture", "body": pad(S, "DLP architecture", 15)},
            {"title": "DLP Policy Framework", "body": numbered_list(S, [
                "Define data types to protect (PII, PCI, PHI, IP) with regular expressions and fingerprinting.",
                "Configure endpoint DLP agent to monitor file copy, print, and USB transfer activities.",
                "Implement email DLP to inspect outbound messages for sensitive content.",
                "Deploy network DLP at egress points to monitor web upload and FTP transfers.",
                "Configure CASB for SaaS application DLP across cloud services.",
                "Set alert thresholds and response actions (monitor, block, quarantine, encrypt).",
                "Establish escalation procedures for DLP violations by severity.",
                "Review and update DLP policies quarterly based on incident data and regulatory changes.",
            ])},
            {"title": "DLP KPIs", "body": kpi_table(S,
                "Table 3.1 – DLP Program KPIs",
                ["KPI", "Target", "Current", "Trend"],
                [
                    ["DLP Policy Coverage (%)", "100%", "94.3%", "Improving"],
                    ["Critical Data Incidents (monthly)", "<5", "2", "Stable"],
                    ["Blocked Exfiltration Attempts", "N/A", "1,847", "Tracking"],
                    ["False Positive Rate (%)", "<20%", "18.3%", "Improving"],
                    ["Avg Investigation Time (hrs)", "<4", "2.8", "Improving"],
                ]
            ) + extra.get((2, 2), [])},
        ]},
        {"title": "Encryption Standards", "sections": [
            {"title": "Encryption Requirements by Data State", "body": kpi_table(S,
                "Table 4.1 – Encryption Standards Matrix",
                ["Data State", "Minimum Standard", "Key Length", "Protocol"],
                [
                    ["Data at Rest", "AES-256", "256-bit", "AES-GCM"],
                    ["Data in Transit", "TLS 1.2+", "2048-bit RSA", "TLS 1.3 preferred"],
                    ["Data in Use", "TEE/SGX where available", "N/A", "Intel SGX/AMD SEV"],
                    ["Backups", "AES-256", "256-bit", "AES-CBC or GCM"],
                    ["Email", "S/MIME or PGP", "2048-bit", "S/MIME preferred"],
                    ["Database", "Transparent Data Encryption", "256-bit", "AES-256-TDE"],
                ]
            )},
            {"title": "Key Management", "body": pad(S, "encryption key management", 15)},
            {"title": "Certificate Lifecycle", "body": pad(S, "certificate lifecycle", 15)},
        ]},
        {"title": "Privacy Governance", "sections": [
            {"title": "Privacy by Design", "body": pad(S, "privacy by design", 15)},
            {"title": "Data Subject Rights Management", "body": bullet_list(S, [
                "Right of Access: Provide data subjects with copies of their personal data within 30 days.",
                "Right to Rectification: Correct inaccurate personal data within 30 days.",
                "Right to Erasure: Delete personal data upon valid request, subject to legal retention requirements.",
                "Right to Restriction: Restrict processing of personal data under specified circumstances.",
                "Right to Portability: Provide personal data in machine-readable format upon request.",
                "Right to Object: Honor objections to processing for marketing or profiling purposes.",
            ])},
            {"title": "Data Protection Impact Assessments", "body": pad(S, "DPIA process", 15)},
        ]},
        {"title": "Data Retention and Disposal", "sections": [
            {"title": "Retention Schedule", "body": kpi_table(S,
                "Table 6.1 – Data Retention Schedule",
                ["Data Category", "Retention Period", "Legal Basis", "Disposal Method"],
                [
                    ["Employee Records", "7 years post-termination", "Tax/Employment law", "Secure deletion"],
                    ["Financial Records", "7 years", "SOX/Tax regulations", "Secure destruction"],
                    ["Customer PII", "Duration of relationship + 5yr", "Contract/Legal", "Anonymization"],
                    ["Security Logs", "7 years", "SOX/Regulatory", "Encrypted archive"],
                    ["Marketing Data", "2 years from consent", "GDPR consent", "Secure deletion"],
                    ["Health Records (PHI)", "6 years minimum", "HIPAA", "DoD 5220.22-M wipe"],
                ]
            )},
            {"title": "Secure Disposal Procedures", "body": pad(S, "secure data disposal", 15)},
            {"title": "Legal Hold Management", "body": pad(S, "legal hold management", 15)},
        ]},
        {"title": "Incident Response for Data Breaches", "sections": [
            {"title": "Breach Detection and Notification", "body": [
                Paragraph("Data breach response is governed by a dedicated playbook that ensures regulatory notification timelines are met. GDPR requires notification to supervisory authorities within 72 hours of becoming aware of a breach. HIPAA requires notification to HHS and affected individuals within 60 days.", S["body"]),
            ] + pad(S, "breach notification procedures", 15)},
            {"title": "Breach Investigation", "body": pad(S, "breach investigation process", 15)},
            {"title": "Regulatory Reporting", "body": pad(S, "regulatory breach reporting", 15)},
        ]},
        {"title": "Governance and Accountability", "sections": [
            {"title": "Data Protection Officer Role", "body": pad(S, "DPO role and responsibilities", 15)},
            {"title": "Third-Party Data Processing", "body": pad(S, "third-party data processing", 15)},
            {"title": "Program Metrics and Reporting", "body": kpi_table(S,
                "Table 8.1 – Data Protection Program KPIs",
                ["KPI", "Target", "Current", "Notes"],
                [
                    ["Data Classification Coverage (%)", "95%", "87.4%", "Improving"],
                    ["DSR Response On-Time (%)", "100%", "96.2%", "Within regulation"],
                    ["DPIAs Completed", "All new projects", "94%", "Minor gaps"],
                    ["Data Breach Notifications (YTD)", "N/A", "1", "GDPR filed"],
                    ["Third-Party DPA Coverage (%)", "100%", "91.8%", "In progress"],
                ]
            )},
        ]},
    ]


def doc_dp(S, title, subtitle, doc_id, extra=None):
    return assemble_doc(S, title, subtitle, doc_id, _dp_chapters(S, title, extra))

# ---------------------------------------------------------------------------
# Document Registry
# ---------------------------------------------------------------------------

def get_all_documents(S):
    return [
        # IAM  (iam_rbac + iam_access_cert have inline charts already)
        ("iam_rbac_policy.pdf",           doc_iam_rbac(S)),
        ("iam_pam_privileged_access.pdf", doc_iam_pam(S)),
        ("iam_access_certification.pdf",  doc_iam_access_cert(S)),
        ("iam_identity_governance.pdf",   doc_iam_iga(S)),
        ("iam_sso_federation.pdf",        doc_iam_sso(S)),
        # Fraud – transaction_monitoring and case_management get charts
        ("fraud_transaction_monitoring.pdf", doc_fraud(S,
            "Transaction Monitoring Framework", "Internal Fraud Prevention", "FRAUD-POL-001",
            extra={(6, 0): bar_chart(S,
                "Figure 7.1 – Monthly Transaction Anomaly Volume by Category",
                ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
                [
                    ("High-Value Outliers",  [42, 38, 55, 61, 49, 53]),
                    ("After-Hours Txns",     [18, 22, 19, 31, 27, 24]),
                    ("Structuring Alerts",   [9,  11,  8, 14, 12, 10]),
                ],
            )}
        )),
        ("fraud_behavioral_analytics.pdf",   doc_fraud(S, "Behavioral Analytics for Fraud Detection", "Internal Fraud Prevention", "FRAUD-POL-002")),
        ("fraud_insider_threat.pdf",         doc_fraud(S, "Insider Threat Detection and Response Program", "Internal Fraud Prevention", "FRAUD-POL-003")),
        ("fraud_anomaly_detection.pdf",      doc_fraud(S, "Anomaly Detection Methodology for Fraud", "Internal Fraud Prevention", "FRAUD-POL-004")),
        ("fraud_case_management.pdf",        doc_fraud(S,
            "Fraud Case Management and Investigation Framework", "Internal Fraud Prevention", "FRAUD-POL-005",
            extra={(2, 2): pie_chart(S,
                "Figure 3.1 – Fraud Case Distribution by Typology (FY 2025)",
                ["Asset Misappropriation", "Financial Statement", "Corruption", "Payroll", "Expense", "Billing"],
                [38, 12, 18, 10, 14, 8],
            )}
        )),
        # Vulnerability Management
        ("vuln_cvss_scoring.pdf",         doc_vuln(S,
            "CVSS Scoring and Vulnerability Rating Framework", "Vulnerability Management", "VULN-POL-001",
            extra={(2, 0): pie_chart(S,
                "Figure 3.1 – Open Vulnerability Distribution by CVSS Severity (Current)",
                ["Critical", "High", "Medium", "Low", "Informational"],
                [87, 312, 1104, 2341, 876],
            )}
        )),
        ("vuln_patch_management.pdf",     doc_vuln(S,
            "Enterprise Patch Management Policy", "Vulnerability Management", "VULN-POL-002",
            extra={(4, 0): bar_chart(S,
                "Figure 5.1 – Patch Compliance % by Platform (Q1 2026)",
                ["Windows Server", "Linux", "macOS", "Network Devices", "Containers", "Cloud VMs"],
                [
                    ("Compliant %",    [96.2, 91.4, 88.7, 79.3, 94.1, 97.8]),
                    ("SLA Target %",   [98,   95,   92,   90,   95,   98]),
                ],
            )}
        )),
        ("vuln_remediation_sla.pdf",      doc_vuln(S,
            "Vulnerability Remediation SLA Framework", "Vulnerability Management", "VULN-POL-003",
            extra={(4, 1): line_chart(S,
                "Figure 5.2 – SLA Compliance Trend by Severity (Quarterly)",
                ["Q1 25", "Q2 25", "Q3 25", "Q4 25", "Q1 26"],
                [
                    ("Critical SLA %", [88, 91, 93, 95, 97.2]),
                    ("High SLA %",     [82, 85, 87, 90, 91.4]),
                    ("Medium SLA %",   [91, 93, 94, 95, 96.1]),
                ],
            )}
        )),
        ("vuln_asset_prioritization.pdf", doc_vuln(S, "Asset-Based Vulnerability Prioritization Framework", "Vulnerability Management", "VULN-POL-004")),
        ("vuln_scanning_framework.pdf",   doc_vuln(S, "Vulnerability Scanning and Assessment Framework", "Vulnerability Management", "VULN-POL-005")),
        # UEBA
        ("ueba_baseline_profiling.pdf",  doc_ueba(S, "User and Entity Baseline Profiling Framework", "UEBA", "UEBA-POL-001")),
        ("ueba_anomaly_detection.pdf",   doc_ueba(S,
            "UEBA Anomaly Detection Methodology", "UEBA", "UEBA-POL-002",
            extra={(2, 0): line_chart(S,
                "Figure 3.1 – UEBA Model Precision and Recall (6-Month Trend)",
                ["Aug 25", "Sep 25", "Oct 25", "Nov 25", "Dec 25", "Jan 26"],
                [
                    ("Precision %", [61, 64, 67, 70, 72, 74]),
                    ("Recall %",    [55, 58, 61, 64, 67, 69]),
                    ("F1 Score %",  [58, 61, 64, 67, 69, 71]),
                ],
            )}
        )),
        ("ueba_risk_scoring.pdf",        doc_ueba(S,
            "UEBA Risk Scoring and Prioritization Framework", "UEBA", "UEBA-POL-003",
            extra={(3, 0): bar_chart(S,
                "Figure 4.1 – Active Entity Count by Risk Score Tier (Jan 2026)",
                ["Low (0-39)", "Medium (40-59)", "High (60-79)", "Critical (80-89)", "Imminent (90-100)"],
                [("Entity Count", [4821, 312, 87, 23, 4])],
            )}
        )),
        ("ueba_alert_triage.pdf",        doc_ueba(S, "UEBA Alert Triage and Investigation Procedures", "UEBA", "UEBA-POL-004")),
        ("ueba_threat_hunting.pdf",      doc_ueba(S,
            "Threat Hunting with UEBA: Methodology and Playbooks", "UEBA", "UEBA-POL-005",
            extra={(6, 0): pie_chart(S,
                "Figure 7.1 – UEBA-Detected Threat Categories (FY 2025)",
                ["Credential Abuse", "Data Exfiltration", "Lateral Movement", "Privilege Escalation", "Policy Violation", "Insider Fraud"],
                [28, 22, 17, 15, 11, 7],
            )}
        )),
        # ASM
        ("asm_asset_inventory.pdf",         doc_asm(S, "Attack Surface Asset Inventory Framework", "Attack Surface Management", "ASM-POL-001")),
        ("asm_exposure_scoring.pdf",        doc_asm(S,
            "Exposure Scoring and Risk Quantification", "Attack Surface Management", "ASM-POL-002",
            extra={(2, 0): bar_chart(S,
                "Figure 3.1 – Open Exposures by Severity and Asset Category",
                ["Web Apps", "Cloud", "Network", "Certificates", "DNS/Domains", "Third-Party"],
                [
                    ("Critical",  [7,  12, 3,  2,  1,  4]),
                    ("High",      [18, 34, 11, 5,  3,  9]),
                    ("Medium",    [42, 67, 28, 12, 8,  21]),
                ],
            )}
        )),
        ("asm_external_attack_surface.pdf", doc_asm(S, "External Attack Surface Management Program", "Attack Surface Management", "ASM-POL-003")),
        ("asm_cloud_asset_discovery.pdf",   doc_asm(S,
            "Cloud Asset Discovery and Shadow IT Detection", "Attack Surface Management", "ASM-POL-004",
            extra={(3, 2): pie_chart(S,
                "Figure 4.1 – Managed Cloud Assets by Provider (Q1 2026)",
                ["AWS", "Azure", "GCP", "Alibaba Cloud", "Other SaaS"],
                [41, 35, 14, 5, 5],
            )}
        )),
        ("asm_remediation_tracking.pdf",    doc_asm(S, "ASM Remediation Tracking and Closure Framework", "Attack Surface Management", "ASM-POL-005")),
        # SIEM
        ("siem_log_aggregation.pdf",  doc_siem(S, "SIEM Log Aggregation and Normalization Framework", "SIEM", "SIEM-POL-001")),
        ("siem_alert_correlation.pdf",doc_siem(S,
            "SIEM Alert Correlation and Threat Detection", "SIEM", "SIEM-POL-002",
            extra={(3, 0): bar_chart(S,
                "Figure 4.1 – Monthly SIEM Alert Volume by Priority (Jan–Jun 2026)",
                ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
                [
                    ("P1 Critical", [12, 9,  14, 11, 8,  10]),
                    ("P2 High",     [78, 82, 91, 74, 88, 79]),
                    ("P3 Medium",   [210, 198, 234, 187, 221, 203]),
                ],
            )}
        )),
        ("siem_use_case_library.pdf", doc_siem(S,
            "SIEM Use Case Library and Detection Engineering", "SIEM", "SIEM-POL-003",
            extra={(2, 1): bar_chart(S,
                "Figure 3.1 – MITRE ATT&CK Tactic Coverage by Detection Status",
                ["Recon", "Resource Dev", "Initial Access", "Execution", "Persistence",
                 "Priv Esc", "Defense Evasion", "Cred Access", "Discovery", "Lateral Mvmt",
                 "Collection", "C2", "Exfiltration", "Impact"],
                [
                    ("Covered %",    [80, 60, 90, 85, 75, 70, 65, 80, 88, 72, 68, 74, 78, 55]),
                    ("In Progress %",[10, 25, 5,  10, 15, 20, 20, 10, 8,  18, 22, 16, 12, 30]),
                ],
            )}
        )),
        ("siem_tuning_framework.pdf", doc_siem(S,
            "SIEM Tuning Framework and False Positive Reduction", "SIEM", "SIEM-POL-004",
            extra={(5, 2): line_chart(S,
                "Figure 6.1 – False Positive Rate Reduction by Use Case Category (Quarterly)",
                ["Q1 25", "Q2 25", "Q3 25", "Q4 25", "Q1 26"],
                [
                    ("Identity/Auth Rules", [72, 65, 58, 52, 44]),
                    ("Network Rules",       [68, 61, 55, 49, 41]),
                    ("Endpoint Rules",      [81, 74, 67, 59, 51]),
                ],
            )}
        )),
        ("siem_incident_response.pdf",doc_siem(S, "SIEM-Driven Incident Response Procedures", "SIEM", "SIEM-POL-005")),
        # Data Protection
        ("dp_dlp_framework.pdf",        doc_dp(S,
            "Data Loss Prevention (DLP) Framework", "Data Protection", "DP-POL-001",
            extra={(2, 2): pie_chart(S,
                "Figure 3.1 – DLP Policy Violations by Data Channel (FY 2025)",
                ["Email", "Endpoint/USB", "Cloud Upload", "Web/HTTP", "Printing", "IM/Collaboration"],
                [34, 22, 19, 13, 7, 5],
            )}
        )),
        ("dp_encryption_standards.pdf", doc_dp(S, "Enterprise Encryption Standards and Key Management", "Data Protection", "DP-POL-002")),
        ("dp_data_classification.pdf",  doc_dp(S,
            "Data Classification and Handling Policy", "Data Protection", "DP-POL-003",
            extra={(1, 0): bar_chart(S,
                "Figure 2.1 – Data Classification Coverage by Business Unit (%)",
                ["Finance", "HR", "Legal", "IT", "Marketing", "Operations", "Sales"],
                [
                    ("Classified %",       [96, 91, 88, 99, 74, 82, 78]),
                    ("Correctly Labeled %",[89, 85, 82, 97, 68, 76, 72]),
                ],
            )}
        )),
        ("dp_privacy_governance.pdf",   doc_dp(S, "Privacy Governance and Compliance Framework", "Data Protection", "DP-POL-004")),
        ("dp_data_retention_policy.pdf",doc_dp(S, "Data Retention and Disposal Policy", "Data Protection", "DP-POL-005")),
    ]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    S = build_styles()

    print("Building document content...")
    all_docs = get_all_documents(S)
    print(f"Generating {len(all_docs)} PDF documents into '{OUTPUT_DIR}/'...\n")

    for i, (filename, elems) in enumerate(all_docs, 1):
        filepath = OUTPUT_DIR / filename
        print(f"  [{i:02d}/{len(all_docs)}] {filename}", end=" ... ", flush=True)
        try:
            doc = SimpleDocTemplate(
                str(filepath),
                pagesize=LETTER,
                rightMargin=0.85 * inch,
                leftMargin=0.85 * inch,
                topMargin=1.0 * inch,
                bottomMargin=1.0 * inch,
            )
            doc.build(elems)
            size_kb = filepath.stat().st_size // 1024
            print(f"OK ({size_kb} KB)")
        except Exception as e:
            print(f"FAILED: {e}")
            raise

    print(f"\nDone. {len(all_docs)} PDFs saved to '{OUTPUT_DIR}/'.")
    _clear_query_cache()


def _clear_query_cache(host: str = "http://localhost:8000") -> None:
    """
    Notify the running RAG server to clear its in-memory query cache.
    Safe to call even when the server is not running — failure is logged, not raised.
    """
    url = f"{host}/cache/clear"
    req = urllib.request.Request(url, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            print(f"\nQuery cache cleared on server ({resp.status}). "
                  "Fresh retrieval will run on the next request.")
    except urllib.error.URLError:
        print(
            "\nNote: RAG server not reachable — query cache was not cleared.\n"
            "      Stale results will expire naturally (TTL) or on next server restart."
        )


if __name__ == "__main__":
    main()
