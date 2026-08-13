"""Build a ten page applied AI project report."""

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
OUTPUT = ROOT / "reports/Consumer_Grievance_Router_India_Report.pdf"
NAVY = colors.HexColor("#20324c")
PURPLE = colors.HexColor("#6d4fa3")
PALE = colors.HexColor("#f1edf8")


def footer(canvas, document):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#667085"))
    canvas.drawString(2 * cm, 1.15 * cm, "Consumer Grievance Router India")
    canvas.drawRightString(19 * cm, 1.15 * cm, f"Page {document.page}")
    canvas.restoreState()


def build_report() -> Path:
    metrics = json.loads((ROOT / "data/processed/evaluation.json").read_text())
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CoverTitle", parent=styles["Title"], fontSize=27, leading=33, textColor=NAVY, alignment=TA_CENTER, spaceAfter=18))
    styles.add(ParagraphStyle(name="SectionTitle", parent=styles["Heading1"], fontSize=20, leading=25, textColor=NAVY, spaceAfter=14))
    styles.add(ParagraphStyle(name="Subhead", parent=styles["Heading2"], fontSize=12, leading=16, textColor=PURPLE, spaceBefore=9, spaceAfter=5))
    styles.add(ParagraphStyle(name="BodyReport", parent=styles["BodyText"], fontSize=10, leading=15, textColor=colors.HexColor("#343b48"), spaceAfter=9))
    doc = SimpleDocTemplate(str(OUTPUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm, title="Consumer Grievance Router India", author="Harika", subject="Applied AI and natural language processing project report")
    story = []
    standard_table = TableStyle([("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d7d0e4")), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("PADDING", (0, 0), (-1, -1), 7)])

    story.extend([Spacer(1, 3.1 * cm), Paragraph("Consumer Grievance<br/>Router India", styles["CoverTitle"]), Paragraph("Interpretable text classification with a manual review path", ParagraphStyle(name="CoverSub", parent=styles["BodyReport"], fontSize=14, leading=20, textColor=PURPLE, alignment=TA_CENTER)), Spacer(1, 1.3 * cm), Table([["Project type", "Applied AI and NLP"], ["Context", "Indian consumer service grievances"], ["Model", "TF-IDF with logistic regression"], ["Prepared by", "Harika"]], colWidths=[4 * cm, 9 * cm], style=TableStyle([("BACKGROUND", (0, 0), (0, -1), PALE), ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"), ("TEXTCOLOR", (0, 0), (0, -1), NAVY), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d7d0e4")), ("PADDING", (0, 0), (-1, -1), 9)])), Spacer(1, 1.1 * cm), Paragraph("This report explains the use case, data design, language pipeline, evaluation, abstention rule, safety boundaries, limitations, and reproducibility.", styles["BodyReport"]), PageBreak()])

    sections = [
        ("1. Executive summary", ["The project routes a short English grievance to Ecommerce, Banking, Telecom, Travel, or Appliances. It does not file a case or decide whether a complaint is valid.", "The model reaches {:.1%} accuracy and {:.3f} macro F1 on a held-out synthetic sample. Because the examples are templated, the result demonstrates pipeline correctness rather than real-world readiness.".format(metrics["accuracy"], metrics["macro_f1"]), "A confidence threshold adds a manual review outcome. This is important because a plausible category can still be wrong when a message combines sectors or uses unfamiliar language."]),
        ("2. Problem and root cause", ["Consumer problems are written in natural language while operational queues use structured categories. Requiring a person to understand the taxonomy adds friction and inconsistent routing.", "The root cause is not only vocabulary. One grievance can mention a bank payment and an ecommerce refund, while a keyword rule may select whichever term appears first.", "The project uses weighted language features, calibrated class probabilities, and an explicit review option. It remains a triage aid rather than a decision maker."]),
        ("3. Dataset design", ["Four hundred fictional English grievances are generated with a fixed seed. Five sectors contribute equal numbers of examples so the training task is not dominated by one class.", "Templates vary items, amounts, plans, speeds, appliances, prefixes, and status descriptions. No actual complaint records, names, phone numbers, account details, or order identifiers are used.", "The main limitation is that templated text is more regular than real messages. The data card therefore prevents the evaluation from being confused with field performance."]),
        ("4. Language pipeline", ["TF-IDF converts unigrams and bigrams into weighted features. Frequent generic words contribute less while useful phrases such as pending refund, ATM cash, broadband speed, and warranty period can influence the class.", "Logistic regression produces one probability per category and keeps the model easier to inspect than a large language model. Class balancing is enabled and the random state is fixed.", "The fitted vectorizer and classifier are saved together. Inference normalises whitespace, rejects extremely short input, checks the highest probability, and returns manual review below the threshold."]),
    ]
    for title, paragraphs in sections:
        story.append(Paragraph(title, styles["SectionTitle"]))
        for paragraph in paragraphs:
            story.append(Paragraph(paragraph, styles["BodyReport"]))
        if title.startswith("3."):
            story.append(Table([["Category", "Example issue"], ["Ecommerce", "Delivery, return, or marketplace refund"], ["Banking", "Debit, ATM, UPI, fee, or card reversal"], ["Telecom", "Recharge, broadband, billing, or porting"], ["Travel", "Train, flight, bus, hotel, or portal booking"], ["Appliances", "Warranty, installation, repair, or replacement"]], colWidths=[4 * cm, 10 * cm], style=standard_table))
        if title.startswith("4."):
            story.append(Table([["Stage", "Control"], ["Input", "Trim and minimum detail check"], ["Features", "TF-IDF unigrams and bigrams"], ["Classifier", "Balanced logistic regression"], ["Confidence", "Highest class probability"], ["Fallback", "Manual review below 0.62"]], colWidths=[4 * cm, 10 * cm], style=standard_table))
        story.append(PageBreak())

    story.extend([Paragraph("5. Evaluation", styles["SectionTitle"]), Image(str(ROOT / "evidence/routing_evaluation.png"), width=17 * cm, height=6.07 * cm), Spacer(1, 0.4 * cm), Table([["Metric", "Result"], ["Rows", str(metrics["rows"])], ["Test rows", str(metrics["test_rows"])], ["Accuracy", f"{metrics['accuracy']:.1%}"], ["Macro F1", f"{metrics['macro_f1']:.3f}"], ["Mean confidence", f"{metrics['mean_confidence']:.3f}"], ["Review threshold", f"{metrics['review_threshold']:.2f}"]], colWidths=[8 * cm, 5 * cm], style=standard_table), Paragraph("Every held-out template variant was classified correctly. This perfect result is a warning about the regular synthetic data, not a reason to claim production quality. A realistic evaluation needs independently written multilingual grievances.", styles["BodyReport"]), PageBreak()])

    final_sections = [
        ("6. Confidence and abstention", ["The router compares the largest class probability with 0.62. A lower score returns manual review and includes the suggested category only as supporting information.", "A threshold moves some errors from automatic routing into a review queue, but it cannot detect every confidently wrong prediction. Threshold selection must use representative validation data and the cost of each misroute.", "Short text is rejected before prediction because phrases such as not working do not identify a sector. Compound grievances should also be reviewed or split into separate issues."]),
        ("7. Explainability and testing", ["The linear classifier allows words and phrases to be inspected by category. This supports debugging when a queue is selected for the wrong reason.", "Tests check balanced data generation, a clear ecommerce example, short-input review, and forced review at a strict threshold. Continuous integration recreates the sample and model before running tests.", "Explainability does not make a model correct. It helps a reviewer understand likely signals and identify missing or misleading examples."]),
        ("8. Safety and use boundaries", ["The router never claims legal authority and does not decide compensation, eligibility, liability, or consumer rights. It does not submit data to the National Consumer Helpline.", "The interface should avoid collecting account numbers, card details, passwords, one-time passwords, or unnecessary identity data. Logs need retention limits and access controls if real text is ever used.", "Users should receive the official helpline link and a chance to correct the selected category before any external action."]),
        ("9. Limitations and next steps", ["The model is English-only, synthetic, and limited to five categories. It does not handle code-mixed language, spelling variation, sarcasm, multiple issues, attachments, or new service sectors reliably.", "Next steps are to collect consented de-identified examples, add Hindi and selected regional languages, measure errors by language and sector, calibrate probabilities, test out-of-scope detection, and conduct manual review studies.", "The repository is reproducible: generation, training, evidence, saved metrics, tests, and report creation are versioned. The README contains the exact commands and the model card defines appropriate use."]),
    ]
    for title, paragraphs in final_sections:
        story.append(Paragraph(title, styles["SectionTitle"]))
        for paragraph in paragraphs:
            story.append(Paragraph(paragraph, styles["BodyReport"]))
        if title.startswith("8."):
            story.append(Table([["Risk", "Control"], ["Ambiguous message", "Manual review"], ["Insufficient detail", "Minimum input check"], ["Overclaiming authority", "Educational scope statement"], ["Sensitive data", "Do not request or store it"], ["Category error", "User correction before action"]], colWidths=[6 * cm, 8 * cm], style=standard_table))
        if not title.startswith("9."):
            story.append(PageBreak())

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return OUTPUT


if __name__ == "__main__":
    print(build_report())

