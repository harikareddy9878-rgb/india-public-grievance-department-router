"""Build the ten-page CivicRoute project report."""

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
OUTPUT = ROOT / "reports/CivicRoute_Service_Request_Triage_Report.pdf"
NAVY = colors.HexColor("#20324c")
PURPLE = colors.HexColor("#6d4fa3")
PALE = colors.HexColor("#f1edf8")


def footer(canvas, document):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#667085"))
    canvas.drawString(2 * cm, 1.1 * cm, "CivicRoute Service Request Triage")
    canvas.drawRightString(19 * cm, 1.1 * cm, f"Page {document.page}")
    canvas.restoreState()


def build_report() -> Path:
    metrics = json.loads((ROOT / "data/processed/evaluation.json").read_text())
    manifest = json.loads((ROOT / "data/raw/source_manifest.json").read_text())
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CoverTitle", parent=styles["Title"], fontSize=29, leading=34, textColor=NAVY, alignment=TA_CENTER, spaceAfter=18))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading1"], fontSize=19, leading=24, textColor=NAVY, spaceAfter=13))
    styles.add(ParagraphStyle(name="BodyR", parent=styles["BodyText"], fontSize=10, leading=15, textColor=colors.HexColor("#343b48"), spaceAfter=9))
    doc = SimpleDocTemplate(str(OUTPUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm, title="CivicRoute Service Request Triage", author="Harika")
    story = []
    table_style = TableStyle([("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d7d0e4")), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("PADDING", (0, 0), (-1, -1), 7)])
    story.extend([Spacer(1, 3 * cm), Paragraph("CivicRoute<br/>Service Request Triage", styles["CoverTitle"]), Paragraph("Interpretable NLP, harder holdout evaluation, and manual review", ParagraphStyle(name="CoverSub", parent=styles["BodyR"], fontSize=14, leading=20, textColor=PURPLE, alignment=TA_CENTER)), Spacer(1, 1.2 * cm), Table([["Project type", "Applied NLP"], ["Source records", f"{metrics['source_rows']:,}"], ["Prepared rows", f"{metrics['model_rows']:,}"], ["Prepared by", "Harika"]], colWidths=[4 * cm, 9 * cm], style=TableStyle([("BACKGROUND", (0, 0), (0, -1), PALE), ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d7d0e4")), ("PADDING", (0, 0), (-1, -1), 9)])), PageBreak()])

    sections = [
        ("1. Executive summary", [f"CivicRoute prepares {metrics['source_rows']:,} source records into {metrics['model_rows']:,} labelled and deduplicated service requests across eight queues.", f"A standard random holdout reaches {metrics['macro_f1']:.3f} macro F1. A harder split that excludes complete category codes from training reaches {metrics['category_holdout_macro_f1']:.3f} macro F1 on {metrics['category_holdout_rows']:,} requests.", f"At the {metrics['review_threshold']:.2f} threshold, the harder split routes {metrics['category_holdout_automatic_coverage']:.1%} automatically and accepted suggestions reach {metrics['category_holdout_automatic_accuracy']:.1%} accuracy in this snapshot."]),
        ("2. Problem and purpose", ["Citizens describe service problems in free text while service teams organise work into structured queues. That mismatch can create avoidable transfers and slow the first response.", "The project asks whether a compact, interpretable model can suggest a useful first queue without hiding uncertainty.", "The output is a triage aid. It does not file a complaint, validate facts, determine priority or legal rights, or replace an official portal."]),
        ("3. Source and privacy boundary", ["The source is the Government of India complaint corpus published on Kaggle as a grievance report. The dataset page states an MIT licence and provides a category mapping workbook.", "Grievance is the source system's formal term for a complaint or service request. CivicRoute uses clearer service-request language in its interface while preserving the original term in provenance notes.", "The source file is named no_pii_grievance.json, but complaint text can still contain details entered by a citizen. Prepared text is therefore not published in Git or exposed as a searchable corpus."]),
        ("4. Preparation pipeline", [f"The acquisition script maps category codes to organisations, selects eight well-represented service queues, removes insufficient text, cleans repeated redaction markers, and drops exact duplicate complaint text. {manifest['rows_with_selected_service_queues']:,} rows pass selection before deduplication.", "The model table keeps a project identifier, request text, queue, organisation code, category code, and category description. Generated raw and prepared text files stay outside Git.", "Queue counts are imbalanced, so the classifier uses class weights and evaluation reports macro F1 alongside accuracy."]),
    ]
    for title, paragraphs in sections:
        story.append(Paragraph(title, styles["Section"]))
        for paragraph in paragraphs:
            story.append(Paragraph(paragraph, styles["BodyR"]))
        if title.startswith("4."):
            story.append(Table([["Service queue", "Prepared rows"]] + [[name, f"{count:,}"] for name, count in manifest["service_queue_counts"].items()], colWidths=[9 * cm, 4 * cm], style=table_style))
        story.append(PageBreak())

    story.extend([Paragraph("5. Evaluation dashboard", styles["Section"]), Image(str(ROOT / "evidence/routing_evaluation.png"), width=17 * cm, height=6.8 * cm), Spacer(1, 0.35 * cm), Table([["Evaluation", "Rows", "Macro F1"], ["Random holdout", f"{metrics['test_rows']:,}", f"{metrics['macro_f1']:.3f}"], ["Category-group holdout", f"{metrics['category_holdout_rows']:,}", f"{metrics['category_holdout_macro_f1']:.3f}"]], colWidths=[7 * cm, 3 * cm, 3 * cm], style=table_style), Paragraph("The confusion matrix shows performance across all eight queues. The confidence plot shows which requests are accepted and which are retained for manual review.", styles["BodyR"]), PageBreak()])

    final = [
        ("6. Language model", ["TF-IDF represents unigrams and bigrams while reducing the influence of generic repeated words. The vocabulary is capped so the complete workflow remains practical on a student laptop.", "Class-balanced logistic regression returns one probability per service queue and allows influential terms to be inspected. Text preparation and classification are stored as one reproducible pipeline.", "This design is smaller, cheaper, and easier to explain than using a general-purpose language model for a fixed eight-queue task."], None),
        ("7. Confidence and manual review", [f"Requests below {metrics['review_threshold']:.2f} confidence are not routed automatically. The best suggestion remains visible as supporting information for a reviewer.", "Messages shorter than 25 characters are sent to review before model inference. Multi-topic, emergency, personally identifying, and unfamiliar requests should also be reviewed.", "A confidence threshold cannot detect every confidently wrong result. It must be recalibrated when queues, vocabulary, or source systems change."], [["Harder holdout measure", "Result"], ["Automatic coverage", f"{metrics['category_holdout_automatic_coverage']:.1%}"], ["Accepted-route accuracy", f"{metrics['category_holdout_automatic_accuracy']:.1%}"], ["Unseen category codes", f"{metrics['category_holdout_categories']}"]]),
        ("8. Evaluation interpretation", ["The random split is useful for comparison but can be optimistic when training and test rows share administrative category language.", "The category-group holdout keeps complete category codes out of training. Its lower score is a more credible measure of how the model handles less familiar request types and is therefore the headline result.", "The corpus is still English-dominant and historical. Future evaluation should include time-based splits, near-duplicate detection, independently written samples, and multilingual requests."], None),
        ("9. Governance, reproducibility, and next steps", ["The repository documents source, licence, privacy boundary, included queues, both evaluation splits, threshold, limitations, and the official CPGRAMS link.", "Acquisition, preparation, training, inference, evidence generation, tests, and this report are versioned. The README provides the full rebuild sequence.", "Next steps are queue-specific thresholds, drift monitoring, multilingual evaluation, a reviewer interface, emergency-message detection, and strict retention controls for request text."], [["Risk", "Control"], ["Wrong automatic queue", "Confidence threshold and human review"], ["Overstated evaluation", "Category-group holdout as headline result"], ["Privacy exposure", "Generated request text excluded from Git"], ["Taxonomy drift", "Version mappings and reevaluate"]]),
    ]
    for title, paragraphs, rows in final:
        story.append(Paragraph(title, styles["Section"]))
        for paragraph in paragraphs:
            story.append(Paragraph(paragraph, styles["BodyR"]))
        if rows:
            story.append(Table(rows, colWidths=[6 * cm, 8 * cm], style=table_style))
        if not title.startswith("9."):
            story.append(PageBreak())
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return OUTPUT


if __name__ == "__main__":
    print(build_report())
