from __future__ import annotations

from io import BytesIO
from typing import Any


def excel_report(metrics: dict[str, Any], enquiries: list[dict[str, Any]]) -> bytes:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill

    workbook = Workbook()
    summary = workbook.active
    summary.title = "Summary"
    summary.append(["Metric", "Value"])
    for key, value in metrics.items():
        summary.append([key.replace("_", " ").title(), value])
    sheet = workbook.create_sheet("Enquiries")
    sheet.append(["Reference", "Category", "Department", "Language", "Status", "Created"])
    for item in enquiries:
        sheet.append([
            item.get("reference"), item.get("category"), item.get("department"),
            item.get("language"), item.get("status"), str(item.get("created_at", "")),
        ])
    for page in workbook.worksheets:
        page.freeze_panes = "A2"
        for cell in page[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="164D3A")
        for column in page.columns:
            width = min(max(max(len(str(cell.value or "")) for cell in column) + 2, 12), 36)
            page.column_dimensions[column[0].column_letter].width = width
    output = BytesIO()
    workbook.save(output)
    return output.getvalue()


def pdf_report(metrics: dict[str, Any], enquiries: list[dict[str, Any]]) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    output = BytesIO()
    document = SimpleDocTemplate(output, pagesize=landscape(A4), rightMargin=16 * mm, leftMargin=16 * mm)
    styles = getSampleStyleSheet()
    story = [Paragraph("Ministry enquiry report", styles["Title"]), Spacer(1, 8 * mm)]
    story.append(Table([[key.replace("_", " ").title(), str(value)] for key, value in metrics.items()], colWidths=[60 * mm, 35 * mm]))
    story.extend([Spacer(1, 8 * mm), Paragraph("Recent enquiries", styles["Heading2"])])
    rows = [["Reference", "Category", "Department", "Language", "Status", "Created"]]
    rows.extend([
        [str(item.get(key, ""))[:80] for key in ("reference", "category", "department", "language", "status", "created_at")]
        for item in enquiries[:100]
    ])
    table = Table(rows, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#164d3a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d5ddd7")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f5f2")]),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(table)
    document.build(story)
    return output.getvalue()