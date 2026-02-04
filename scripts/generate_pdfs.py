#!/usr/bin/env python3
"""
Professional PDF Generator for CV and Cover Letter
Raiffeisen Research - Research Innovation Hub Application
Clean, professional formatting
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, cm
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import black, white, gray
import os

# Colors - Professional Blue Theme
PRIMARY_BLUE = HexColor('#1a365d')  # Dark navy
ACCENT_BLUE = HexColor('#2c5282')   # Medium blue
TEXT_DARK = HexColor('#1a202c')     # Almost black
TEXT_GRAY = HexColor('#4a5568')     # Dark gray
DIVIDER_GRAY = HexColor('#cbd5e0')  # Light gray for lines

# Page settings
PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN = 2 * cm

def create_styles():
    """Create all paragraph styles for the documents."""
    styles = {}

    # Name/Header style
    styles['Name'] = ParagraphStyle(
        'Name',
        fontName='Helvetica-Bold',
        fontSize=24,
        textColor=PRIMARY_BLUE,
        spaceAfter=2*mm,
        alignment=TA_LEFT,
    )

    # Contact info style
    styles['Contact'] = ParagraphStyle(
        'Contact',
        fontName='Helvetica',
        fontSize=9,
        textColor=TEXT_GRAY,
        spaceAfter=3*mm,
        leading=12,
    )

    # Section header style
    styles['SectionHeader'] = ParagraphStyle(
        'SectionHeader',
        fontName='Helvetica-Bold',
        fontSize=10,
        textColor=PRIMARY_BLUE,
        spaceBefore=4*mm,
        spaceAfter=2*mm,
    )

    # Subsection header (job title/company)
    styles['SubsectionHeader'] = ParagraphStyle(
        'SubsectionHeader',
        fontName='Helvetica-Bold',
        fontSize=9.5,
        textColor=TEXT_DARK,
        spaceBefore=2.5*mm,
        spaceAfter=0.5*mm,
    )

    # Date/location style
    styles['DateLocation'] = ParagraphStyle(
        'DateLocation',
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        textColor=TEXT_GRAY,
        spaceAfter=1*mm,
    )

    # Bullet point style
    styles['Bullet'] = ParagraphStyle(
        'Bullet',
        fontName='Helvetica',
        fontSize=8.5,
        textColor=TEXT_DARK,
        leading=11,
        leftIndent=3*mm,
        bulletIndent=0,
        spaceAfter=0.5*mm,
    )

    # Skills style
    styles['Skills'] = ParagraphStyle(
        'Skills',
        fontName='Helvetica',
        fontSize=8.5,
        textColor=TEXT_DARK,
        leading=11,
        spaceAfter=1*mm,
    )

    # Profile summary style
    styles['Profile'] = ParagraphStyle(
        'Profile',
        fontName='Helvetica',
        fontSize=9,
        textColor=TEXT_DARK,
        leading=12,
        spaceAfter=2*mm,
        alignment=TA_JUSTIFY,
    )

    # Cover letter styles
    styles['CoverDate'] = ParagraphStyle(
        'CoverDate',
        fontName='Helvetica',
        fontSize=10,
        textColor=TEXT_GRAY,
        spaceBefore=4*mm,
        spaceAfter=4*mm,
        alignment=TA_LEFT,
    )

    styles['CoverSubject'] = ParagraphStyle(
        'CoverSubject',
        fontName='Helvetica-Bold',
        fontSize=11,
        textColor=PRIMARY_BLUE,
        spaceBefore=2*mm,
        spaceAfter=4*mm,
    )

    styles['CoverBody'] = ParagraphStyle(
        'CoverBody',
        fontName='Helvetica',
        fontSize=10,
        textColor=TEXT_DARK,
        leading=13,
        spaceAfter=2.5*mm,
        alignment=TA_JUSTIFY,
    )

    styles['CoverSalutation'] = ParagraphStyle(
        'CoverSalutation',
        fontName='Helvetica',
        fontSize=10,
        textColor=TEXT_DARK,
        spaceBefore=3*mm,
        spaceAfter=1*mm,
    )

    styles['Signature'] = ParagraphStyle(
        'Signature',
        fontName='Helvetica-Bold',
        fontSize=10,
        textColor=TEXT_DARK,
        spaceBefore=5*mm,
    )

    return styles

def create_divider():
    """Create a horizontal divider line."""
    return HRFlowable(
        width="100%",
        thickness=0.5,
        color=DIVIDER_GRAY,
        spaceBefore=0.5*mm,
        spaceAfter=1.5*mm
    )

def create_section_divider():
    """Create a section divider with accent color."""
    return HRFlowable(
        width="100%",
        thickness=1,
        color=PRIMARY_BLUE,
        spaceBefore=0,
        spaceAfter=2*mm
    )

def build_cv():
    """Build the CV PDF - Professional formatting."""
    output_path = "/Users/moe/projects/Master-Thesis/cv_tailored_polished.pdf"
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=1.5*cm,
        bottomMargin=1.2*cm
    )

    styles = create_styles()
    story = []

    # Header - Name with M.Sc.
    story.append(Paragraph("Markus Öffel, M.Sc.", styles['Name']))
    story.append(Spacer(1, 2*mm))

    # Contact info - clean format
    contact = """<font color="#4a5568">Ausstellungsstrasse 55/14, 1020 Vienna, Austria</font><br/>
<font color="#2c5282">+43 650 2300064</font> &nbsp;&nbsp;|&nbsp;&nbsp; <font color="#2c5282">markus.oeffel@gmail.com</font>"""
    story.append(Paragraph(contact, styles['Contact']))

    story.append(create_section_divider())

    # Profile Summary
    story.append(Paragraph("PROFILE", styles['SectionHeader']))
    profile_text = """Quantitative Finance and Data Science professional specializing in time series analysis,
    volatility forecasting, and risk modeling. Hands-on experience building quantitative models
    in Python and R for forecasting, portfolio analysis, and asset allocation.
    Proven track record in cross-functional product ownership, automated reporting, and data-driven decision support.
    Seeking to develop innovative quantitative models and digital research products for retail and private banking clients."""
    story.append(Paragraph(profile_text, styles['Profile']))

    # Core Skills
    story.append(Paragraph("CORE COMPETENCIES", styles['SectionHeader']))
    story.append(create_divider())

    skills_data = [
        ["<b>Quantitative Methods:</b>", "Time Series Analysis (ARIMA, GARCH/EGARCH/FIGARCH), Monte Carlo Simulation, Value-at-Risk (VaR), Expected Shortfall (ES), Forecasting Models"],
        ["<b>Portfolio Analytics:</b>", "Portfolio Analysis, Asset Allocation, Equity Scoring, Minimum-Variance Optimization, Sharpe Ratio Optimization"],
        ["<b>Risk Analysis:</b>", "Market Risk, Interest Rate Risk, Credit Risk Assessment, Model Backtesting, Kupiec Test, Christoffersen Test"],
        ["<b>Programming:</b>", "Python (pandas, NumPy, statsmodels, arch, scikit-learn), R, SQL, Excel/VBA"],
        ["<b>AI / Machine Learning:</b>", "LLM Engineering, RAG (Retrieval-Augmented Generation), QLoRA Fine-Tuning, AI Agents"],
        ["<b>Reporting & Visualization:</b>", "Jupyter Notebooks, R Markdown, Power BI, Tableau, Automated Reports, Dashboards"],
        ["<b>Collaboration & Agile:</b>", "Git, Atlassian (Jira, Confluence), Scrum, Agile, Cross-functional Stakeholder Management"],
    ]

    for skill_row in skills_data:
        skill_text = f"{skill_row[0]} {skill_row[1]}"
        story.append(Paragraph(skill_text, styles['Skills']))

    # Experience
    story.append(Paragraph("PROFESSIONAL EXPERIENCE", styles['SectionHeader']))
    story.append(create_divider())

    # Job 1
    story.append(Paragraph("<b>Product Owner / Data & Financing</b>", styles['SubsectionHeader']))
    story.append(Paragraph("Swisslife Select Austria — Vienna/Hybrid | Jul 2025 – Present", styles['DateLocation']))

    bullets_1 = [
        "Partner with Head of Sales on strategic data initiatives, developing KPIs and forecasting models for business planning",
        "Serve as Product Owner for financing squad, aligning advisor requirements with business needs and IT delivery in Agile/Scrum environment",
        "Analyze sales and financing data using Python and SQL; deliver automated management reports, forecasts, and decision templates",
        "Core member of internal AI taskforce: design and prioritize automation and AI-driven use cases for operational efficiency",
        "Contributor to Investment Squad: support asset allocation discussions and portfolio analysis with quantitative input",
    ]
    for b in bullets_1:
        story.append(Paragraph(f"• {b}", styles['Bullet']))

    # Job 2
    story.append(Paragraph("<b>Financing Specialist</b>", styles['SubsectionHeader']))
    story.append(Paragraph("Swisslife Select Austria — Graz/Hybrid | May 2024 – Jun 2025", styles['DateLocation']))

    bullets_2 = [
        "Advised asset managers and clients on mortgage financing strategies, applying market data analysis and interest rate risk assessment",
        "Analyzed credit risk and market risk in private real-estate financing portfolios",
        "Developed decision templates based on interest rate forecasts and regulatory requirements",
    ]
    for b in bullets_2:
        story.append(Paragraph(f"• {b}", styles['Bullet']))

    # Job 3
    story.append(Paragraph("<b>Project Manager, Network & Events</b>", styles['SubsectionHeader']))
    story.append(Paragraph("Verein Netzwerk Logistik — Kapfenberg | Feb 2023 – Oct 2023", styles['DateLocation']))

    bullets_3 = [
        "Built and maintained cross-industry corporate network with 50+ member companies",
        "Planned, budgeted, and executed industry events and workshops; delivered presentations and stakeholder communications",
    ]
    for b in bullets_3:
        story.append(Paragraph(f"• {b}", styles['Bullet']))

    # Job 4
    story.append(Paragraph("<b>Quality Manager</b>", styles['SubsectionHeader']))
    story.append(Paragraph("Hanfama Pflanzen Produktions GmbH — Graz | Sep 2020 – Jan 2023", styles['DateLocation']))

    bullets_4 = [
        "Optimized quality management system and process documentation",
        "Served as interface between purchasing, sales, production, and management",
    ]
    for b in bullets_4:
        story.append(Paragraph(f"• {b}", styles['Bullet']))

    # Projects / Research
    story.append(Paragraph("RESEARCH & PROJECTS", styles['SectionHeader']))
    story.append(create_divider())

    story.append(Paragraph("<b>Master's Thesis: ARIMA-GARCH Models for Cryptocurrency Risk Assessment</b>", styles['SubsectionHeader']))
    story.append(Paragraph("University of Graz | 2025", styles['DateLocation']))

    thesis_bullets = [
        "Built end-to-end Python pipeline for time series forecasting and volatility modeling across multiple asset classes (BTC, ETH, DOGE, SOL)",
        "Implemented ARIMA + GARCH/EGARCH/FIGARCH variants with automated model selection using AIC/BIC criteria",
        "Developed rolling backtest framework with multi-horizon evaluation for forecasting model validation",
        "Conducted risk evaluation via Value-at-Risk (VaR) and Expected Shortfall (ES) with Kupiec and Christoffersen backtesting",
        "Created automated reporting system generating CSV exports, visualization plots, and reproducible Jupyter Notebooks",
    ]
    for b in thesis_bullets:
        story.append(Paragraph(f"• {b}", styles['Bullet']))

    # Education
    story.append(Paragraph("EDUCATION", styles['SectionHeader']))
    story.append(create_divider())

    story.append(Paragraph("<b>M.Sc. Business Administration</b> — University of Graz", styles['SubsectionHeader']))
    story.append(Paragraph("Oct 2023 – Oct 2025 | Focus: Corporate Finance, Investments, Business Analytics & Data Science", styles['DateLocation']))
    story.append(Paragraph("• Specialization in quantitative investment methods: minimum-variance optimization, Sharpe ratio optimization, Monte Carlo techniques", styles['Bullet']))
    story.append(Paragraph("• Coursework in portfolio analysis, asset allocation, and financial econometrics", styles['Bullet']))

    story.append(Paragraph("<b>B.Sc. Innovation Management</b> (part-time) — Campus02 UAS Graz", styles['SubsectionHeader']))
    story.append(Paragraph("Sep 2020 – Jul 2023", styles['DateLocation']))

    story.append(Paragraph("<b>Higher Technical Institute</b> (Industrial Engineering, Logistics) — Leoben", styles['SubsectionHeader']))
    story.append(Paragraph("2004 – 2009", styles['DateLocation']))

    # Certifications
    story.append(Paragraph("CERTIFICATIONS & UPCOMING EXAMS", styles['SectionHeader']))
    story.append(create_divider())

    cert_text = """<b>CFA Level I Exam</b> — scheduled for <b>19 August 2026</b><br/>
AI Engineer Core Track: LLM Engineering, RAG, QLoRA, Agents (Udemy) — Jan 2026<br/>
Google Advanced Data Analytics Professional Certificate (Coursera) — Feb 2025<br/>
DataCamp: Data Analyst in Python — Feb 2025<br/>
DataCamp: Python Data Fundamentals — Feb 2025"""
    story.append(Paragraph(cert_text, styles['Skills']))

    # Languages
    story.append(Paragraph("LANGUAGES", styles['SectionHeader']))
    story.append(create_divider())
    story.append(Paragraph("German (C2 Native) &nbsp;|&nbsp; English (C1 Fluent) &nbsp;|&nbsp; French (A2 Basic)", styles['Skills']))

    # Build PDF
    doc.build(story)
    print(f"CV saved to: {output_path}")
    return output_path

def build_cover_letter():
    """Build the Cover Letter PDF - fits on one page."""
    output_path = "/Users/moe/projects/Master-Thesis/cover_letter_polished.pdf"
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=1.8*cm,
        bottomMargin=1.5*cm
    )

    styles = create_styles()
    story = []

    # Header - Name with M.Sc.
    story.append(Paragraph("Markus Öffel, M.Sc.", styles['Name']))
    story.append(Spacer(1, 2*mm))

    # Contact info
    contact = """<font color="#4a5568">Ausstellungsstrasse 55/14, 1020 Vienna, Austria</font><br/>
<font color="#2c5282">+43 650 2300064</font> &nbsp;&nbsp;|&nbsp;&nbsp; <font color="#2c5282">markus.oeffel@gmail.com</font>"""
    story.append(Paragraph(contact, styles['Contact']))

    story.append(create_section_divider())

    # Date
    story.append(Paragraph("Vienna, 4 February 2026", styles['CoverDate']))

    # Subject line
    story.append(Paragraph("Application — Research Innovation Hub, Raiffeisen Research", styles['CoverSubject']))

    # Salutation
    story.append(Paragraph("Dear Raiffeisen Research Team,", styles['CoverBody']))

    # Body paragraphs - compact for one page
    para1 = """I am writing to express my strong interest in the Research Innovation Hub position.
    The opportunity to develop quantitative models for asset class analysis, equity scoring,
    portfolio analysis, and asset allocation—while transforming these into accessible digital
    research products for retail and private banking clients—aligns precisely with my professional
    goals and technical expertise."""
    story.append(Paragraph(para1, styles['CoverBody']))

    para_values = """What particularly attracts me to Raiffeisen Research is your commitment to turning
    continuous innovation into superior customer experience. I am drawn to the collaborative approach
    between your Vienna-based team and the 13 CEE country units, which enables you to deliver timely,
    actionable research that genuinely serves client needs. This customer-centric philosophy resonates
    with my own professional values."""
    story.append(Paragraph(para_values, styles['CoverBody']))

    para2 = """In my master's thesis at the University of Graz, I built an end-to-end Python pipeline
    evaluating ARIMA-GARCH model variants for cryptocurrency risk assessment across multiple
    asset classes (BTC, ETH, DOGE, SOL). This work involved automated model selection using AIC/BIC
    criteria, rolling backtests with multi-horizon evaluation, and risk quantification via
    Value-at-Risk (VaR) and Expected Shortfall (ES) with Kupiec and Christoffersen backtesting.
    The project demonstrates my ability to implement robust time series and volatility
    forecasting models—competencies directly applicable to your team's quantitative research initiatives."""
    story.append(Paragraph(para2, styles['CoverBody']))

    para3 = """At Swisslife Select, I currently serve as Product Owner for the financing squad,
    bridging business requirements with IT delivery in an Agile/Scrum environment while analyzing
    sales and financing data using Python and SQL to produce management reports and forecasts.
    As a member of both the internal AI taskforce and the Investment Squad, I contribute to
    asset allocation discussions and design data-driven use cases. This cross-functional experience
    mirrors the collaborative structure you describe for the Research Innovation Hub."""
    story.append(Paragraph(para3, styles['CoverBody']))

    para4 = """I bring hands-on experience with Jupyter Notebooks and R Markdown for
    automated reporting and dashboard creation, working knowledge of Git and
    Atlassian workflows (Jira, Confluence), and strong Scrum practices from my product
    ownership role. I am comfortable working with market data from various sources and preparing
    results for presentations and stakeholder communications in both German and English."""
    story.append(Paragraph(para4, styles['CoverBody']))

    para5 = """To further strengthen my quantitative finance credentials, I am scheduled to sit the
    <b>CFA Level I exam on 19 August 2026</b>. Additionally, I have recently completed the
    AI Engineer Core Track covering LLM Engineering, RAG, and AI Agents—skills I believe will
    become increasingly valuable for developing next-generation research products."""
    story.append(Paragraph(para5, styles['CoverBody']))

    para6 = """I am excited about the prospect of contributing to Raiffeisen Research's mission of
    delivering innovative, data-driven research products that create real value for clients.
    I would welcome the opportunity to discuss how my background in quantitative modeling,
    risk analysis, portfolio analysis, and cross-functional product development can support
    your team's objectives."""
    story.append(Paragraph(para6, styles['CoverBody']))

    # Closing
    story.append(Paragraph("Sincerely,", styles['CoverSalutation']))
    story.append(Paragraph("Markus Öffel", styles['Signature']))

    # Build PDF
    doc.build(story)
    print(f"Cover Letter saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    print("Generating professional PDF documents...")
    print("-" * 50)
    cv_path = build_cv()
    cover_path = build_cover_letter()
    print("-" * 50)
    print("Done! Generated files:")
    print(f"  1. {cv_path}")
    print(f"  2. {cover_path}")
