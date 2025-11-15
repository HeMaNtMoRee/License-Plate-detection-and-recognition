import json
import re
from typing import Dict, Optional, Any
from rapidfuzz import process, fuzz

# ---------------- Data Mappings ---------------- #
UAE_IDENTIFIERS = {
    # Abu Dhabi
    **{str(n): "Abu Dhabi" for n in [1, 50] + list(range(4, 22))},

    # Dubai
    **{c: "Dubai" for c in list("ABCDEFGHJKLMNPRSTUVWXYZ")},
    **{cc: "Dubai" for cc in ["AA", "BB", "CC", "DD"]},

    # Sharjah
    **{str(n): "Sharjah" for n in range(1, 5)},

    # Ajman
    **{c: "Ajman" for c in list("ABCDEH")},

    # Fujairah
    **{c: "Fujairah" for c in list("ABCDEFGKMPRST")},

    # Ras Al Khaimah
    **{c: "Ras Al Khaimah" for c in ["A", "C", "D", "I", "K", "M", "N", "S", "V", "Y"]},

    # Umm Al Quwain
    **{c: "Umm Al Quwain" for c in list("ABCDEFGHIX")},
}

UAE_REGIONS_FULL = {
    "ABU DHABI": "Abu Dhabi",
    "DUBAI": "Dubai",
    "SHARJAH": "Sharjah",
    "AJMAN": "Ajman",
    "FUJAIRAH": "Fujairah",
    "RAS AL KHAIMAH": "Ras Al Khaimah",
    "UMM AL QUWAIN": "Umm Al Quwain",
    "AD": "Abu Dhabi",
    "RAK": "Ras Al Khaimah",
    "UAQ": "Umm Al Quwain"
}

# uae_regions = [
#     "Abu Dhabi",
#     "Dubai",
#     "Sharjah",
#     "Ajman",
#     "Fujairah",
#     "Ras Al Khaimah",
#     "Umm Al Quwain"
# ]

def extract_category(texts, region, number):
    for text in texts:
        if text == number:
            continue

        # Abu Dhabi → numeric code 1, 4–21, 50 OR "AD"
        if region == "Abu Dhabi":
            if text == "AD" or text.isdigit() and (text == "1" or 4 <= int(text) <= 21 or text == "50"):
                return text

        # Dubai → single/double letters
        elif region == "Dubai":
            if re.fullmatch(r"[A-Z]{1,2}", text) and text not in {"I", "O", "Q"}:
                return text

        # Sharjah → SHJ or digits 1–5
        elif region == "Sharjah":
            if  text.isdigit() and 1 <= int(text) <= 5:
                return text

        # Ajman → "AJ" or letters A, B, C, D, E, H
        elif region == "Ajman":
            if  text in {"A", "B", "C", "D", "E", "H"}:
                return text

        # Ras Al Khaimah → "RAK" or letters A, C, D, I, K, M, N, S, V, Y
        elif region == "Ras Al Khaimah":
            if text in {"A", "C", "D", "I", "K", "M", "N", "S", "V", "Y"}  :
                return text

        # Fujairah → "FUJ" or letters A–G, K, M, P, R, S, T
        elif region == "Fujairah":
            if text in {"A", "B", "C", "D", "E", "F", "G", "K", "M", "P", "R", "S", "T"} :
                return text

        # Umm Al Quwain → "UAQ" or letters A, B, C, D, E, F, G, H, I, X
        elif region == "Umm Al Quwain":
            if text in {"A", "B", "C", "D", "E", "F", "G", "H", "I", "X"} :
                return text

    return None

def autocorrect_region(token, threshold=70):
    """
    Autocorrect a single OCR token to closest UAE region using fuzzy matching.
    Returns the corrected region or original token if no close match.
    """
    token_clean = token.strip().title()  # normalize case

    # Use token_sort_ratio for better handling of OCR issues
    match, score, _ = process.extractOne(
        token_clean,
        UAE_REGIONS_FULL.values(),
        scorer=fuzz.token_sort_ratio
    )

    # Debugging print
    # print(f"Token: {token_clean} -> Match: {match}, Score: {score}")

    if score >= threshold:
        return match
    return token 

def format_license_plate(data: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """Extracts and formats UAE license plate data from OCR result."""
    

    rec_texts = [
        part
        for t in data.get("rec_texts", [])
        if isinstance(t, str) and t.strip()
        for word in t.strip().upper().split()
        for part in (
            [autocorrect_region(word).upper()]
            if re.match(r"^([A-ZΑ-ΩΡ]\.){2,}[A-Z0-9]*$", autocorrect_region(word).upper())  # abbreviation → keep whole
            else autocorrect_region(word).upper().split(".")  # normal case → split
        )
        if part  # avoid empty strings
    ]





   

    print("Texts:", rec_texts)
    # print("Scores:",data.get("rec_scores",[]))

    region_name = None
    number = None
    category = None

    # 1. Match full emirate name
    for text in rec_texts:
        if text in UAE_REGIONS_FULL:
            region_name = UAE_REGIONS_FULL[text]
            # print("uppppppppp")
            break

    # 2. If not found, check identifier mapping
    if not region_name:
        for text in rec_texts:
            if text in UAE_IDENTIFIERS:
                # print("dddddddd")
                region_name = UAE_IDENTIFIERS[text]
                category = text  # If identifier is like AD, SHJ, or single letter, treat as category
                break

    # 3. Extract number (1–5 digits)
    max_len = 0
    for text in rec_texts:
        if re.fullmatch(r"\d{1,5}", text):  # matches only 1–5 digit numbers
            if len(text) >= max_len:  # keep the longest
                number = text
                max_len = len(text)

    # 4. Extract category if not already set (numeric or single letter, excluding main number)
  
    if not category:
        category=extract_category(rec_texts,region_name,number)
        

    # 5. Build license plate string
    license_plate = None
    if category and region_name and number:
        license_plate = f"{category} {region_name} {number}"
    elif region_name and number:
        license_plate = f"{region_name} {number}"
    # print(f"license plate: {license_plate},category: {category},region: {region_name},number: {number}")

    return {
        "license plate": license_plate,
        "category": category,
        "region": region_name,
        "number": number
    }



def process_json_data(ocr_data: dict) -> Dict[str, Optional[str]]:
    """Processes OCR data from JSON/dict and returns formatted result."""
    return format_license_plate(ocr_data)


