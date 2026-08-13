"""Build a ten page report for the CPGRAMS department router."""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/India_Public_Grievance_Department_Router_Report.pdf"
NAVY = colors.HexColor("#20324c")
PURPLE = colors.HexColor("#6d4fa3")
PALE = colors.HexColor("#f1edf8")


def footer(canvas, document):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#667085"))
    canvas.drawString(2 * cm, 1.1 * cm, "India Public Grievance Department Router")
    canvas.drawRightString(19 * cm, 1.1 * cm, f"Page {document.page}")
    canvas.restoreState()


def build_report() -> Path:
    metrics = json.loads((ROOT / "data/processed/evaluation.json").read_text())
    manifest = json.loads((ROOT / "data/raw/source_manifest.json").read_text())
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CoverTitle", parent=styles["Title"], fontSize=27, leading=33, textColor=NAVY, alignment=TA_CENTER, spaceAfter=18))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading1"], fontSize=19, leading=24, textColor=NAVY, spaceAfter=13))
    styles.add(ParagraphStyle(name="Sub", parent=styles["Heading2"], fontSize=12, leading=16, textColor=PURPLE, spaceBefore=8, spaceAfter=5))
    styles.add(ParagraphStyle(name="BodyR", parent=styles["BodyText"], fontSize=10, leading=15, textColor=colors.HexColor("#343b48"), spaceAfter=9))
    doc = SimpleDocTemplate(str(OUTPUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm, title="India Public Grievance Department Router", author="Harika")
    story = []
    table_style = TableStyle([("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d7d0e4")), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("PADDING", (0, 0), (-1, -1), 7)])
    story.extend([Spacer(1, 3.0 * cm), Paragraph("India Public Grievance<br/>Department Router", styles["CoverTitle"]), Paragraph("Interpretable NLP with confidence and manual review", ParagraphStyle(name="CoverSub", parent=styles["BodyR"], fontSize=14, leading=20, textColor=PURPLE, alignment=TA_CENTER)), Spacer(1, 1.2 * cm), Table([["Project type", "Applied NLP"], ["Source records", f"{metrics['source_rows']:,}"], ["Model rows", f"{metrics['model_rows']:,}"], ["Prepared by", "Harika"]], colWidths=[4 * cm, 9 * cm], style=TableStyle([("BACKGROUND", (0, 0), (0, -1), PALE), ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d7d0e4")), ("PADDING", (0, 0), (-1, -1), 9)])), PageBreak()])

    sections = [
        ("1. Executive summary", [f"The source contains {metrics['source_rows']:,} de-identified grievance records. Eight departments with sufficient mapped labels contribute {metrics['model_rows']:,} deduplicated model rows.", f"The held-out result is {metrics['accuracy']:.1%} accuracy and {metrics['macro_f1']:.3f} macro F1 across {metrics['test_rows']:,} grievances.", f"At a {metrics['review_threshold']:.2f} confidence threshold, {metrics['automatic_coverage']:.1%} of test rows are automatically routed and accepted predictions reach {metrics['automatic_accuracy']:.1%} accuracy in this snapshot."]),
        ("2. Problem and intended use", ["A public grievance is unstructured text while operational ownership is represented by departments and categories. Manual first-level routing can be slow and inconsistent.", "The project asks whether a compact language model can provide a useful department suggestion without hiding uncertainty.", "The output is a routing aid only. It does not submit a grievance, validate facts, decide eligibility, determine legal rights, or replace CPGRAMS."]),
        ("3. Source and privacy boundary", ["The Government of India: Grievance report corpus is downloaded from Kaggle. The dataset page states an MIT licence and provides a category mapping workbook.", "The source file used here is explicitly named no_pii_grievance.json. This project does not use complainant attributes, does not reconstruct removed values, and does not attempt re-identification.", "Complaint text may still contain personal details entered by a citizen. The project should therefore be treated as controlled educational data and not republished as a searchable public corpus."]),
        ("4. Preparation pipeline", [f"The acquisition script maps category codes to organisation codes, selects eight departments, removes records without mapped labels or sufficient text, cleans repeated redaction markers, and drops duplicate complaint text. {manifest['rows_with_selected_labeled_departments']:,} rows pass selection before text deduplication.", "The model table retains a project grievance identifier, complaint text, department, organisation code, category code, and category description.", "Department counts remain imbalanced, so the classifier uses class weights and the evaluation reports macro F1 alongside accuracy."]),
    ]
    for title, paragraphs in sections:
        story.append(Paragraph(title, styles["Section"]))
        for paragraph in paragraphs:
            story.append(Paragraph(paragraph, styles["BodyR"]))
        if title.startswith("4."):
            story.append(Table([["Department", "Prepared rows"]] + [[name, f"{count:,}"] for name, count in manifest["department_counts"].items()], colWidths=[9 * cm, 4 * cm], style=table_style))
        story.append(PageBreak())

    story.extend([Paragraph("5. Evaluation dashboard", styles["Section"]), Image(str(ROOT / "evidence/routing_evaluation.png"), width=17 * cm, height=6.8 * cm), Spacer(1, 0.35 * cm), Table([["Metric", "Result"], ["Test rows", f"{metrics['test_rows']:,}"], ["Accuracy", f"{metrics['accuracy']:.1%}"], ["Macro F1", f"{metrics['macro_f1']:.3f}"], ["Manual review rows", f"{metrics['manual_review_rows']:,}"], ["Automatic coverage", f"{metrics['automatic_coverage']:.1%}"]], colWidths=[8 * cm, 5 * cm], style=table_style), Paragraph("The normalised confusion matrix makes minority-class performance visible. The confidence histogram shows how the review threshold separates accepted and deferred predictions.", styles["BodyR"]), PageBreak()])

    final = [
        ("6. Language model", ["TF-IDF represents unigrams and bigrams while reducing the weight of generic repeated words. The vocabulary is capped to keep training practical on a student laptop.", "Balanced logistic regression provides one probability per department and allows influential terms to be inspected. The vectorizer and classifier are persisted as one pipeline.", "This approach is intentionally smaller and easier to reproduce than a large language model for the same classification task."]),
        ("7. Confidence and manual review", [f"A message below {metrics['review_threshold']:.2f} confidence is not automatically routed. The response retains the best suggestion so a reviewer can inspect the model's direction without treating it as final.", "Messages shorter than 25 characters are sent to review before the model runs. Compound complaints and unfamiliar organisations should also be reviewed.", "A threshold cannot detect every confidently wrong prediction. It must be retested when category definitions, departments, or language change."]),
        ("8. Evaluation interpretation", ["The very high held-out score is credible for this split but has a specific reason: department and category wording strongly identify many records. Near-duplicate structure may also make randomly held-out text easier than future grievances.", "The model does not cover every organisation or Indian language. It is not evaluated on OCR text, adversarial wording, code switching, or future category changes.", "A stronger follow-up would split by time, test exact and near-duplicate leakage, add Hindi and regional-language evaluation, and involve human review of a blind sample."]),
        ("9. Governance, reproducibility, and next steps", ["The repository documents source, licence, privacy boundary, included departments, metrics, threshold, limitations, and official CPGRAMS link.", "Acquisition, preparation, training, routing, evidence generation, tests, and this report are versioned. The README provides the full rebuild sequence.", "Next steps are a time-aware split, department-specific thresholds, drift monitoring, multilingual evaluation, a review interface, and careful retention controls for complaint text."]),
    ]
    for title, paragraphs in final:
        story.append(Paragraph(title, styles["Section"]))
        for paragraph in paragraphs:
            story.append(Paragraph(paragraph, styles["BodyR"]))
        if title.startswith("9."):
            story.append(Table([["Risk", "Control"], ["Wrong automatic department", "Confidence threshold and human review"], ["Privacy exposure", "Use de-identified source and avoid re-identification"], ["Taxonomy drift", "Version mapping and reevaluate"], ["Misleading use", "Routing aid boundary beside every result"]], colWidths=[5 * cm, 9 * cm], style=table_style))
        else:
            story.append(PageBreak())
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return OUTPUT


if __name__ == "__main__":
    print(build_report())
