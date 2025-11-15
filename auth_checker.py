# auth_checker.py
import json

REGISTER_FILE = "registered_plates/registered_plates.json"

def load_registered_plates():
    with open(REGISTER_FILE, "r") as f:
        return json.load(f)["plates"]

def check_plate_authorization(plate_result: dict) -> str:
    """
    plate_result = {
        "license_plate": "...",
        "category": "...",
        "region": "...",
        "number": "..."
    }
    Returns: "authorized", "blacklisted", or "unauthorized"
    """

    region = plate_result.get("region")
    category = plate_result.get("category")
    number = plate_result.get("number")

    if not region or not number:
        return "unauthorized"   # incomplete OCR → reject

    db = load_registered_plates()

    for entry in db:
        if (
            entry["region"] == region and
            entry["number"] == number and
            entry["category"] == category
        ):
            return entry["status"]

    return "unauthorized"
