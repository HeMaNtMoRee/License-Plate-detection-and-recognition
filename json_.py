import json
import re
from typing import Dict, Optional, Any
from rapidfuzz import process, fuzz
from functools import lru_cache
import time
from datetime import datetime

# ---------------- Data Mappings ---------------- #
UAE_IDENTIFIERS = {
    # Example filled if needed...
}

UAE_REGIONS_FULL = {
    "ABU DHABI": "Abu Dhabi",
    "A.D": "Abu Dhabi",
    "A.D.": "Abu Dhabi",
    "AD": "Abu Dhabi",
    
    "DUBAI": "Dubai",
    
    "SHARJAH": "Sharjah",
    "SHJ": "Sharjah",
    
    "AJMAN": "Ajman",
    
    "FUJAIRAH": "Fujairah",
    "FUJ": "Fujairah",
    
    "RAS AL KHAIMAH": "Ras Al Khaimah",
    "RAK": "Ras Al Khaimah",
    "R.A.K.": "Ras Al Khaimah",
    "R.A.K": "Ras Al Khaimah",
    
    "UMM AL QUWAIN": "Umm Al Quwain",
    "UAQ": "Umm Al Quwain",
    "U.A.Q": "Umm Al Quwain",
    "U.A.Q.": "Umm Al Quwain"

}

# import json
# from datetime import datetime
# from typing import Dict, Any, Optional




def text_(data: Dict[str, Any]):
    try:
        
        current_datetime = datetime.now()
        current_time = current_datetime.time()
        
        ocr_text=data.get("rec_texts", [])
        print(ocr_text)
        with open("ocr.txt", 'a') as f:
            # f.write(json.dumps({"time": current_time, "result": ocr_text}) + "\n")
            f.write(json.dumps({"time": current_time.isoformat(), "result": ocr_text}) + "\n")

    except Exception as e:
        print(e)
        
    


def format_license_plate(data: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """Extracts and formats UAE license plate data from OCR result (optimized)."""
    # print("heeeeeeeeeee")
    
    valid_region= ["A.D",'A.D.','U.A.Q','U.A.Q.',"R.A.K.","R.A.K",]
    
    text_(data)
    print("fff")
    rec_texts = [
        part
        for t in data.get("rec_texts", [])
        if isinstance(t, str) and t.strip()
        for word in t.strip().upper().split()
        for part in (
            (
                # --- Autocorrect inline ---
                (lambda token: (
                    (lambda token_clean: (
                        token_clean if token_clean in UAE_REGIONS_FULL 
                        else (
                            lambda match, score,_: match if score >= 70 else token
                        )(*process.extractOne(token_clean, UAE_REGIONS_FULL.values(), scorer=fuzz.token_sort_ratio))
                    ))(token.strip().title())
                ))(word)
            ).upper()
            ,
        )  # wrap into tuple so we can reuse below
        for part in (
            [part]  # case: abbreviation like U.A.E
            if re.match(r"^([A-ZΑ-ΩΡ]\.){2,}[A-Z0-9]*$", part)
            else [p for p in part.split(".") if p]   # case: dot split (except A.D)
                if "." in part and part not in valid_region
            else list(re.match(r"^([A-Z]+)(\d+)$", part).groups())  # case: alphanum split
                if re.match(r"^([A-Z]+)(\d+)$", part)
            else [part]  # default
        )
        if part
    ]
    
    
    # rec_texts=[]
    # for i in record:
    #     if i.startswith('UAE'):
    #         aa,b,c=i.partition('UAE')
    #         rec_texts.append([b,c])
    #     else:
    #         rec_texts.append(i)    

    print("Texts:", rec_texts)
    # print(data.get("rec_scores", []))

    region_name, number, category = None, None, None
    max_len = 0

    # Single loop → extract region, number, and potential category
    for text in rec_texts:
        if not region_name and text in UAE_REGIONS_FULL:
            region_name = UAE_REGIONS_FULL[text]

        if re.fullmatch(r"\d{1,5}", text):  # number candidate
            if len(text) >= max_len:
                number, max_len = text, len(text)

    if region_name and not category:
        # category = extract_category(rec_texts, region_name, number)
        for text in rec_texts:
            if text == number:
                continue
            if region_name == "Abu Dhabi" and (text == "AD" or (text.isdigit() and (text == "1" or 4 <= int(text) <= 21 or text == "50"))):
                category= text
            if region_name == "Dubai" and (re.fullmatch(r"[A-Z]", text) or (text in ["AA", "BB", "CC", "DD"])) and text not in {"I", "O", "Q"}:
                category= text
            if region_name == "Sharjah" and text.isdigit() and 1 <= int(text) <= 5:
                category= text
            if region_name == "Ajman" and text in {"A", "B", "C", "D", "E", "H"}:
                category= text
            if region_name == "Ras Al Khaimah" and text in {"A", "C", "D", "I", "K", "M", "N", "S", "V", "Y"}:
                category= text
            if region_name == "Fujairah" and text in {"A", "B", "C", "D", "E", "F", "G", "K", "M", "P", "R", "S", "T"}:
                category= text
            if region_name == "Umm Al Quwain" and text in {"A", "B", "C", "D", "E", "F", "G", "H", "I", "X"}:
                category= text

    license_plate = None
    if category and region_name and number:
        license_plate = f"{category} {region_name} {number}"
    elif region_name and number:
        license_plate = f"{region_name} {number}"

    return {"license plate": license_plate, "category": category, "region": region_name, "number": number}



def process_json_data(ocr_data: dict) -> Dict[str, Optional[str]]:
    result = format_license_plate(ocr_data)
    
    return result


