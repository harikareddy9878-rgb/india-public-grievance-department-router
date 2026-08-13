"""Route a grievance or send it to manual review."""

from __future__ import annotations

from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
THRESHOLD = 0.62


def route_grievance(text: str, threshold: float = THRESHOLD) -> dict:
    clean = " ".join(text.split())
    if len(clean) < 12:
        return {"route": "Manual review", "confidence": 0.0, "reason": "Please provide more detail."}
    model = joblib.load(ROOT / "models/grievance_router.joblib")
    probabilities = model.predict_proba([clean])[0]
    best_index = int(probabilities.argmax())
    confidence = float(probabilities[best_index])
    predicted = str(model.classes_[best_index])
    if confidence < threshold:
        return {"route": "Manual review", "suggested_category": predicted, "confidence": round(confidence, 3), "reason": "The text is ambiguous or outside the learned examples."}
    return {"route": predicted, "confidence": round(confidence, 3), "reason": "Highest model probability above the review threshold."}


if __name__ == "__main__":
    print(route_grievance("My phone order was marked delivered but I did not receive it. Please review this."))
