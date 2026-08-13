"""Generate a deterministic synthetic grievance classification dataset."""

from __future__ import annotations

import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/grievances.csv"
TEMPLATES = {
    "Ecommerce": [
        "My {item} order was marked delivered but I did not receive it",
        "The online seller has not processed my refund for the returned {item}",
        "I received a damaged {item} and the shopping app closed my return request",
        "The price changed after checkout for my {item} order",
        "My prepaid {item} order was cancelled but the amount is still not credited",
    ],
    "Banking": [
        "An unauthorised debit of rupees {amount} appeared in my bank account",
        "The ATM did not dispense cash but rupees {amount} was deducted",
        "My UPI transfer failed and the amount has not returned",
        "The bank charged an unexpected fee on my savings account",
        "My card payment was reversed by the merchant but the bank has not credited it",
    ],
    "Telecom": [
        "My mobile data stopped working after I recharged the {plan} plan",
        "The network provider added a service I did not request",
        "I was billed twice for my broadband connection this month",
        "My number port request is delayed without an explanation",
        "The broadband speed is much lower than the promised {speed} Mbps",
    ],
    "Travel": [
        "My train ticket was cancelled but the refund is still pending",
        "The airline changed my flight and did not offer a suitable refund",
        "The bus operator cancelled the trip and kept the booking amount",
        "My hotel booking was not honoured even though it was prepaid",
        "The travel portal charged a cancellation fee that was not shown before booking",
    ],
    "Appliances": [
        "My {appliance} stopped working during the warranty period",
        "The service centre has delayed the repair of my {appliance}",
        "The technician charged me even though my {appliance} is under warranty",
        "A replacement part for my {appliance} has not arrived",
        "The new {appliance} was installed incorrectly and support is not responding",
    ],
}


def generate_rows(seed: int = 9878, variants_per_template: int = 16) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    prefixes = ["Please help.", "I need support because", "Complaint:", "I have been waiting for a resolution.", ""]
    suffixes = ["Please review this.", "I have the transaction details.", "This happened last week.", "Support has not resolved it.", ""]
    for category, templates in TEMPLATES.items():
        for template in templates:
            for _ in range(variants_per_template):
                text = template.format(item=rng.choice(["phone", "shoes", "mixer", "book", "headphones"]), amount=rng.choice([499, 1200, 3500, 8000]), plan=rng.choice(["monthly", "unlimited", "prepaid"]), speed=rng.choice([50, 100, 200]), appliance=rng.choice(["washing machine", "refrigerator", "television", "microwave", "air conditioner"]))
                text = " ".join(part for part in [rng.choice(prefixes), text, rng.choice(suffixes)] if part)
                rows.append({"text": text, "category": category})
    rng.shuffle(rows)
    return rows


def write_data() -> Path:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    rows = generate_rows()
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "category"])
        writer.writeheader()
        writer.writerows(rows)
    return OUTPUT


if __name__ == "__main__":
    print(write_data())

