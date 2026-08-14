"""Route a civic service request to a queue or explicit manual review."""

from __future__ import annotations

from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]


def route_request(text: str, threshold: float = 0.58) -> dict:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) < 25:
        return {"route": "Manual review", "reason": "More detail is required."}
    model = joblib.load(ROOT / "models/civicroute_triage.joblib")
    probability = model.predict_proba([cleaned])[0]
    best = int(probability.argmax())
    suggestion = str(model.classes_[best])
    confidence = float(probability[best])
    if confidence < threshold:
        return {
            "route": "Manual review",
            "suggested_queue": suggestion,
            "confidence": round(confidence, 3),
        }
    return {"route": suggestion, "confidence": round(confidence, 3)}
