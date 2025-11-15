import json
import nltk
import re
import os
from typing import Dict, Optional, Any
import time

# ---------------- Data Mappings ---------------- #
UAE_IDENTIFIERS = {
    # Abu Dhabi
    **{str(n): "Abu Dhabi" for n in [1, 50] + list(range(4, 22))},
    # **{str(n): "Abu Dhabi" for n in list(range(1, 50))},
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


from rapidfuzz import process, fuzz

# UAE region names
uae_regions = ['ABU DHABI',
 'DUBAI',
 'SHARJAH',
 'AJMAN',
 'FUJAIRAH',
 'RAS AL KHAIMAH',
 'UMM AL QUWAIN']



def autocorrect_region(token, threshold=60):
    """
    Autocorrect a single OCR token to closest UAE region using fuzzy matching.
    Returns the corrected region or original token if no close match.
    """
    token_clean = token.strip().title()  # normalize case

    # Use token_sort_ratio for better handling of OCR issues
    match, score, _ = process.extractOne(
        token_clean,
        uae_regions,
        scorer=fuzz.token_sort_ratio
    )

    # Debugging print
    # print(f"Token: {token_clean} -> Match: {match}, Score: {score}")

    if score >= threshold:
        return match
    return token  # return original if no confident match




def format_license_plate(data: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """Extracts and formats UAE license plate data from OCR result."""
    
    rec_texts = [t.strip().upper() for t in data.get("rec_texts", []) if isinstance(t, str) and t.strip()]
    
    rec_texts = [autocorrect_region(token) for token in rec_texts]
    print("Texts:", rec_texts)
    # print("Scores:",data.get("rec_scores",[]))

    region_name = None
    number = None
    category = None

    # 1. Match full emirate name
    for text in rec_texts:
        if text in UAE_REGIONS_FULL:
            region_name = UAE_REGIONS_FULL[text]
            print("uppppppppp")
            break

    # 2. If not found, check identifier mapping
    if not region_name:
        for text in rec_texts:
            if text in UAE_IDENTIFIERS:
                print("dddddddd")
                region_name = UAE_IDENTIFIERS[text]
                category = text  # If identifier is like AD, SHJ, or single letter, treat as category
                break

    # 3. Extract number (1–5 digits)
    # for text in rec_texts:
    #     if re.fullmatch(r"\d{1,5}", text):
    #         number = text
    #         break
    max_len = 0
    for text in rec_texts:
        if re.fullmatch(r"\d{1,5}", text):  # matches only 1–5 digit numbers
            if len(text) >= max_len:  # keep the longest
                number = text
                max_len = len(text)

    # 4. Extract category if not already set (numeric or single letter, excluding main number)
    if not category:
        
        for text in rec_texts:
            if text != number and re.fullmatch(r"(?:[A-Z]{1,2}|\d{1,2})", text):
                category = text
                break

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


# def format_license_plate(data: Dict[str, Any]) -> Dict[str, Optional[str]]:
#     """Extracts and formats license plate data from OCR result."""
#     rec_texts = [t.strip().upper() for t in data.get("rec_texts", []) if isinstance(t, str) and t.strip()]
#     print("Texts:", rec_texts)

#     region_name = None
#     number = None
#     category = None   # new field

#     # 1. Match full emirate name
#     for text in rec_texts:
#         if text in UAE_REGIONS_FULL:
#             region_name = UAE_REGIONS_FULL[text]
#             break

#     # 2. If not found, check identifier mapping
#     if not region_name:
#         for text in rec_texts:
#             if text in UAE_IDENTIFIERS:
#                 region_name = UAE_IDENTIFIERS[text]
#                 break

#     # 3. Extract number (3–5 digits)
#     for text in rec_texts:
#         if re.fullmatch(r"\d{5}", text):
#             number = text
#             break

#     # 4. Extract category (usually 1–2 digits, but not the main number)
#     for text in rec_texts:
#         if (re.fullmatch(r"\d{1,2}", text) and text != number):
#             category = text
#             break

#     # 5. Build license plate string
#     license_plate = None
#     if category and region_name and number:
#         license_plate = f"{category} {region_name} {number}"
#     elif region_name and number:
#         license_plate = f"{region_name} {number}"

#     return {
#         "license plate": license_plate,
#         "category": category,
#         "region": region_name,
#         "number": number
#     }


# ---------------- json Processing ---------------- #
def process_json_data(ocr_data: dict) -> None:
    """Processes OCR data from a variable and prints formatted result."""
    # start_time = time.time()
    formatted = format_license_plate(ocr_data)
    return formatted
    # end_time = time.time()
    # print("total time",end_time-start_time)
    # print("Your number plate is:", formatted)

# with open('img3_res (1).json', "r", encoding="utf-8") as f:
#     ocr_data = json.load(f)
#     # ocr_data=f.read()
#     # print(ocr_data)
#     process_json_data(ocr_data)
    