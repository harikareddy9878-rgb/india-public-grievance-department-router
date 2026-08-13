"""Download and prepare the de-identified Government of India grievance corpus."""

from __future__ import annotations

import io
import json
import re
import ssl
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import certifi
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = "https://www.kaggle.com/api/v1/datasets/download/ayushyajnik/government-of-india-grievance-report"
SOURCE_PAGE = "https://www.kaggle.com/datasets/ayushyajnik/government-of-india-grievance-report"
RAW_OUTPUT = ROOT / "data/raw/cpgrams_grievances.jsonl"
MODEL_OUTPUT = ROOT / "data/processed/grievances.csv"
MANIFEST = ROOT / "data/raw/source_manifest.json"
SELECTED_ORGS = {
    "DOAAC": "Agriculture",
    "MOLBR": "Labour and Employment",
    "CBODT": "Direct Taxes",
    "DEABD": "Financial Services",
    "DOTEL": "Telecommunications",
    "DPOST": "Postal Services",
    "MORLY": "Railways",
    "MORTH": "Road Transport",
}


def clean_text(value: object) -> str:
    cleaned = re.sub(r"X(?:[A-Z]X){2,}", " ", str(value or ""), flags=re.IGNORECASE)
    cleaned = re.sub(r"[-_=]{4,}", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def download() -> zipfile.ZipFile:
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "HarikaGrievanceProject/2.0"})
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(request, timeout=180, context=context) as response:  # noqa: S310
        return zipfile.ZipFile(io.BytesIO(response.read()))


def load_mapping(archive: zipfile.ZipFile) -> dict[int, dict]:
    content = io.BytesIO(archive.read("CategoryCode_Mapping.xlsx"))
    frame = pd.read_excel(content, sheet_name="Complaint Category")
    return frame.set_index("Code").to_dict("index")


def prepare() -> dict:
    archive = download()
    mapping = load_mapping(archive)
    raw_records = json.loads(archive.read("no_pii_grievance.json"))
    RAW_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    prepared = []
    category_counts: Counter[str] = Counter()
    with RAW_OUTPUT.open("w", encoding="utf-8") as raw_handle:
        for record in raw_records:
            raw_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            code = record.get("CategoryV7")
            category = mapping.get(code, {}) if code is not None else {}
            org = str(category.get("OrgCode") or record.get("org_code") or "")
            label = SELECTED_ORGS.get(org)
            complaint = clean_text(record.get("subject_content_text"))
            if not label or len(complaint) < 60 or code is None:
                continue
            prepared.append(
                {
                    "grievance_id": record.get("registration_no"),
                    "text": complaint,
                    "department": label,
                    "org_code": org,
                    "category_code": int(code),
                    "category_description": clean_text(category.get("Description")),
                }
            )
            category_counts[label] += 1
    frame = pd.DataFrame(prepared).drop_duplicates(subset=["text"]).reset_index(drop=True)
    frame.to_csv(MODEL_OUTPUT, index=False)
    summary = {
        "source_rows": len(raw_records),
        "rows_with_selected_labeled_departments": len(prepared),
        "model_rows": len(frame),
        "departments": sorted(SELECTED_ORGS.values()),
        "department_counts": frame["department"].value_counts().to_dict(),
        "source": SOURCE_PAGE,
        "license": "MIT as stated on the Kaggle dataset page",
        "privacy": "The source files are explicitly named no_pii; this project does not attempt re-identification.",
    }
    MANIFEST.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2))
