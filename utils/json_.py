# json_.py
import re
import json
from typing import Dict, Optional, Any, List, Tuple
from difflib import SequenceMatcher, get_close_matches
from datetime import datetime
import os

# ---------------- Log File ----------------
cwd_path = os.getcwd()
log_path = os.path.join(cwd_path, "logs")
os.makedirs(log_path, exist_ok=True)
LOG_FILE = "logs/plate_logs.jsonl"

def log_plate_result(result: dict, status: str): # <-- MODIFICATION 1: Added 'status' argument
    """
    Write a normalized plate result to a log file in JSON format.
    Result must contain:
      - license_plate
      - category
      - region
      - number
      - confidences
    """
    now = datetime.now()
    log_entry = {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "license_plate": result.get("license_plate"),
        "category": result.get("category"),
        "region": result.get("region"),
        "number": result.get("number"),
        "status": status  # <-- MODIFICATION 2: Added 'status' to the log entry
    }

    # Append safely to .jsonl file (1 record per line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

# ---------------- Canonical UAE model ----------------
# Each emirate has canonical name + list of aliases (common OCR variants)
# + a category validation function (returns True if token is a valid category for that emirate).
UAE_CANONICAL = {
    "Abu Dhabi": {
        "aliases": ["ABU DHABI", "ABU-DHABI", "A D", "A.D", "A.D.", "AD"],
        "category_check": lambda t: t.isdigit() and (t == "1" or 4 <= int(t) <= 21 or t == "50"),
    },
    "Dubai": {
        "aliases": ["DUBAI", "DB", "DXB"],
        "category_check": lambda t: (t.isalpha() and len(t) <= 2 and t not in {"I", "O", "Q"}) or (t.isdigit() and len(t) <= 2),
    },
    "Sharjah": {
        "aliases": ["SHARJAH", "SHJ", "SH"],
        "category_check": lambda t: t.isdigit() and 1 <= int(t) <= 5,
    },
    "Ajman": {
        "aliases": ["AJMAN", "AJ"],
        "category_check": lambda t: t in {"A", "B", "C", "D", "E", "H"},
    },
    "Fujairah": {
        "aliases": ["FUJAIRAH", "FUJ", "FUJARAH", "FUJA"],
        "category_check": lambda t: t in {"A","B","C","D","E","F","G","K","M","P","R","S","T"},
    },
    "Ras Al Khaimah": {
        "aliases": ["RAS AL KHAIMAH", "RAK", "R.A.K", "R.A.K."],
        "category_check": lambda t: t in {"A","C","D","I","K","M","N","S","V","Y"},
    },
    "Umm Al Quwain": {
        "aliases": ["UMM AL QUWAIN", "UAQ", "U.A.Q", "U.A.Q."],
        "category_check": lambda t: t in {"A","B","C","D","E","F","G","H","I","X"},
    },
}

# Flatten alias -> canonical mapping for quick exact matching
_ALIAS_TO_REGION = {}
for region, info in UAE_CANONICAL.items():
    for alias in info["aliases"]:
        cleaned = re.sub(r'\s+|\.', '', alias).upper()
        _ALIAS_TO_REGION[cleaned] = region

# ---------------- Utility functions ----------------
def now_iso_time() -> str:
    return datetime.now().isoformat()

def normalize_token(tok: str) -> str:
    """Uppercase, strip, collapse whitespace."""
    return re.sub(r'\s+', ' ', (tok or "").strip()).upper()

def clean_alias_form(tok: str) -> str:
    """Alias canonicalization used for quick alias lookup (remove dots/spaces)."""
    return re.sub(r'\s+|\.', '', normalize_token(tok))

def digits_only(s: str) -> Optional[str]:
    if s is None:
        return None
    s = re.sub(r'\D', '', s)
    return s if s else None

def similar(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

# ---------------- Tokenization & OCR helpers ----------------
SPLIT_RE = re.compile(r"[,\-_/\\\|\:\;\(\)\[\]]+")  # split on common separators

def tokenize_rec_texts(rec_texts: List[str]) -> List[str]:
    """
    Robust token extraction from raw OCR strings.
    Keeps alpha, digits, dotted abbreviations (U.A.Q), and alnum combinations.
    """
    tokens: List[str] = []
    for t in rec_texts:
        if not isinstance(t, str):
            continue
        s = t.strip()
        if not s:
            continue
        # Split on whitespace first, then further split on punctuation while keeping dotted groups
        parts = []
        for part in s.split():
            # preserve dotted abbreviations like U.A.Q or R.A.K.
            if re.fullmatch(r'([A-Za-z]\.){1,4}[A-Za-z]?', part):
                parts.append(part)
                continue
            # split on separators but keep alnum blocks together (e.g., AB123)
            subparts = [p for seg in SPLIT_RE.split(part) for p in seg.split('.') if p]
            parts.extend([p for p in subparts if p])
        for p in parts:
            pnorm = normalize_token(p)
            if pnorm:
                tokens.append(pnorm)
    return tokens

# ---------------- Region detection ----------------
def detect_region_from_tokens(tokens: List[str]) -> Tuple[Optional[str], float]:
    """
    Returns (region_name or None, confidence 0..1)
    Strategy:
    1) Exact alias match (very high confidence)
    2) Clean alias exact match (remove dots/spaces)
    3) Fuzzy match among aliases using difflib (fallback)
    """
    if not tokens:
        return None, 0.0

    # exact alias token
    for tok in tokens:
        tok_clean = clean_alias_form(tok)
        if tok_clean in _ALIAS_TO_REGION:
            return _ALIAS_TO_REGION[tok_clean], 0.99

    # try fuzzy across all alias strings
    # build a candidate list of cleaned aliases
    alias_list = list(_ALIAS_TO_REGION.keys())
    # join tokens to single string to catch phrases like "RAS AL KHAIMAH"
    joined = " ".join(tokens)
    joined_clean = clean_alias_form(joined)
    matches = get_close_matches(joined_clean, alias_list, n=1, cutoff=0.7)
    if matches:
        score = similar(joined_clean, matches[0])
        return _ALIAS_TO_REGION[matches[0]], score

    # try token-level fuzzy matches (best token wins)
    best_region, best_score = None, 0.0
    for tok in tokens:
        tok_clean = clean_alias_form(tok)
        matches = get_close_matches(tok_clean, alias_list, n=1, cutoff=0.6)
        if matches:
            score = similar(tok_clean, matches[0])
            if score > best_score:
                best_score = score
                best_region = _ALIAS_TO_REGION[matches[0]]
    if best_region:
        return best_region, best_score

    return None, 0.0

# ---------------- Category extraction ----------------
def detect_category_for_region(region: str, tokens: List[str]) -> Tuple[Optional[str], float]:
    """
    Try to find a token that fits the category rules for the detected region.
    Returns (category or None, confidence)
    Confidence is heuristic: exact matches -> 0.95, fuzzy or digit heuristics lower.
    """
    if not region or not tokens:
        return None, 0.0
    checker = UAE_CANONICAL.get(region, {}).get("category_check")
    if not checker:
        return None, 0.0

    # Prefer tokens that are short (1-2 chars) and match the rule
    best_tok, best_conf = None, 0.0
    for tok in tokens:
        # ignore plain numerical sequences longer than 5 (likely the plate number)
        if tok.isdigit() and len(tok) > 5:
            continue
        try:
            if checker(tok):
                # prefer letter categories slightly higher than numeric ones (heuristic)
                conf = 0.95 if tok.isalpha() else 0.9
                # If same length as known category patterns, bump confidence slightly
                if len(tok) <= 2:
                    conf += 0.02
                if conf > best_conf:
                    best_conf = conf
                    best_tok = tok
        except Exception:
            continue

    # As fallback: if no category token found but region has known numeric ranges (e.g., Abu Dhabi),
    # try to infer category if a short numeric token appears that fits the numeric rule.
    return (best_tok, best_conf) if best_tok else (None, 0.0)

# ---------------- Number extraction ----------------
def detect_number(tokens: List[str]) -> Tuple[Optional[str], float]:
    """
    Extract the plate's numeric part. Heuristics:
    - prefer longest contiguous digit token (1-5 digits)
    - if token contains letters+digits (e.g., 'AB123'), extract digits
    - return confidence based on token length and purity
    """
    if not tokens:
        return None, 0.0

    best_num, best_conf = None, 0.0
    for tok in tokens:
        digits = digits_only(tok) or ""
        if not digits:
            continue
        # only accept up to 5 digits for UAE common plates
        if 1 <= len(digits) <= 5:
            # longer digit sequences get slightly better confidence
            conf = 0.6 + (len(digits)/5)*0.35  # range roughly 0.6..0.95
            # if token was pure digits, bump confidence
            if tok.isdigit():
                conf += 0.03
            if conf > best_conf:
                best_conf = min(conf, 0.99)
                best_num = digits
    return (best_num, best_conf) if best_num else (None, 0.0)

# ---------------- Main formatting function ----------------
def format_license_plate(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input: { "rec_texts": [...], "rec_scores": [...] }
    Output: normalized dict with license_plate, category, region, number, confidences
    """
    raw_texts = data.get("rec_texts", [])
    raw_scores = data.get("rec_scores", [])
    # Print raw OCR texts and scores
    # print(json.dumps({"time": now_iso_time(), "rec_texts": raw_texts, "rec_scores": raw_scores}, ensure_ascii=False))

    tokens = tokenize_rec_texts(raw_texts)

    region, region_conf = detect_region_from_tokens(tokens)
    number, number_conf = detect_number(tokens)
    category, category_conf = detect_category_for_region(region, tokens) if region else (None, 0.0)

    # If category missing but region is detected and number exists, attempt region-specific heuristics
    if not category and region:
        # Abu Dhabi sometimes encodes category as small numeric tokens near number; check short numeric tokens
        if region == "Abu Dhabi":
            for tok in tokens:
                if tok.isdigit() and len(tok) <= 2:
                    # test category rule
                    if UAE_CANONICAL[region]['category_check'](tok):
                        category, category_conf = tok, 0.75
                        break

    # Compose license_plate string in canonical casing
    license_plate = None
    if category and region and number:
        license_plate = f"{category} {region} {number}"
    elif region and number:
        license_plate = f"{region} {number}"

    # ensure normalized outputs
    out = {
        "license_plate": license_plate,
        "category": category,
        "region": region,
        "number": number,
        # "confidences": {
        #     "region": round(region_conf, 3),
        #     "category": round(category_conf, 3),
        #     "number": round(number_conf, 3),
        # }
    }
    return out

# ---------------- Convenience wrapper ----------------
def process_json_data(ocr_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    External entrypoint
    """
    result = format_license_plate(ocr_data)
    return result