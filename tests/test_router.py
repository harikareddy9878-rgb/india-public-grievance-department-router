from src.generate_data import generate_rows
from src.route import route_grievance


def test_dataset_is_balanced():
    rows = generate_rows(variants_per_template=3)
    counts = {}
    for row in rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    assert len(set(counts.values())) == 1


def test_clear_ecommerce_grievance_routes_correctly():
    result = route_grievance("My phone order was marked delivered but I did not receive it. Please review this.")
    assert result["route"] == "Ecommerce"


def test_short_input_goes_to_review():
    result = route_grievance("not working")
    assert result["route"] == "Manual review"


def test_ambiguous_text_can_be_forced_to_review():
    result = route_grievance("The company has not helped me with this confusing problem for several days", threshold=0.95)
    assert result["route"] == "Manual review"
