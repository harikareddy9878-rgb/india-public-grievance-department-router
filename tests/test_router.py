from scripts.acquire_data import clean_text
from src.route import route_grievance


def test_clean_text_removes_repeated_redaction_markers():
    assert "XAXPX" not in clean_text("Please XAXPX check the pension case")


def test_clear_railway_grievance_routes_correctly():
    result = route_grievance(
        "Indian Railways cancelled my reserved train ticket from Hyderabad to Delhi. IRCTC refund has not reached my bank account after twenty days. Please route this complaint to the Railway Board."
    )
    assert result["route"] == "Railways"


def test_short_input_goes_to_review():
    result = route_grievance("not working")
    assert result["route"] == "Manual review"


def test_strict_threshold_forces_manual_review():
    result = route_grievance(
        "Please review this delayed government service request because I have not received an update.",
        threshold=0.99,
    )
    assert result["route"] == "Manual review"
