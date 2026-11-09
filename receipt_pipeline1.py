"""
Confidence-based receipt data extraction and validation (development prototype)
Elnur Jabbarov, RTU bachelor thesis, supervisor Prof. I. Erins

The code has four parts that are kept separate:
  1. Extraction - OCR + Gemini (one reading from OCR text, one reading from the image)
  2. Validator  - business rules R1-R4, uses only the extracted data
  3. Routing    - confidence signals -> decision for S0, S1, S2, S3
  4. Evaluator  - compares with the CORD ground truth, never used for decisions

All numbers in PROVISIONAL are assumptions for now. They will be calibrated later
on development data, never on the test set.
"""
import json
import re
import os
import time
import hashlib
import copy
from collections import Counter


# ---- settings

AMOUNT_FIELDS = ["subtotal", "tax", "service", "discount", "total"]

PROVISIONAL = {
    "weights_initial": {"cross": 0.4, "evidence": 0.3, "rule": 0.3},
    "weights_after": {"cross": 0.25, "repeat": 0.15, "evidence": 0.3, "rule": 0.3},
    # (minimum confidence, probability of error)
    "p_error_table": [(0.9, 0.005), (0.6, 0.05), (0.3, 0.20), (0.0, 0.50)],
    # cost in EUR if a wrong value of this field is approved
    "error_cost": {"total": 20.0, "subtotal": 10.0, "tax": 10.0, "service": 10.0,
                   "discount": 10.0, "item_price": 2.0},
    "review_cost": 0.40,
    "uncertain_threshold": 0.9,
    "max_reextraction_rounds": 1,
    "reread_call_cost_estimate": 0.001,
}


# ---- normalisation

def normalize_amount(value):
    """'Rp 12.500' or '12,500' or 12500 -> 12500. Empty -> None. '-5,000' or '(5,000)' -> -5000."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(round(value))

    text = str(value).strip()
    if text == "":
        return None

    is_negative = text.startswith("-") or (text.startswith("(") and text.endswith(")"))
    digits = re.sub(r"[^0-9]", "", text)
    if digits == "":
        return None

    number = int(digits)
    if is_negative:
        return -number
    return number


def normalize_qty(value):
    if value is None or str(value).strip() == "":
        return None
    numbers = re.findall(r"\d+", str(value))
    if numbers:
        return int(numbers[0])
    return None


def numbers_in_text(text):
    """All amounts that appear in the OCR text. Used for the evidence signal."""
    found = set()
    for token in re.findall(r"\d[\d.,]*\d|\d", text or ""):
        number = normalize_amount(token)
        if number is not None:
            found.add(number)
    return found


# ---- schema

def empty_record():
    return {"line_items": [], "subtotal": None, "tax": None, "service": None,
            "discount": None, "total": None}


def coerce_record(answer):
    """Put a parsed answer into our fixed schema. A missing field becomes None, never 0."""
    record = empty_record()
    if not isinstance(answer, dict):
        return record

    for field in AMOUNT_FIELDS:
        record[field] = normalize_amount(answer.get(field))
    if record["discount"] is not None:
        record["discount"] = abs(record["discount"])

    items = answer.get("line_items")
    if not isinstance(items, list):
        items = []
    for item in items:
        if not isinstance(item, dict):
            continue
        record["line_items"].append({
            "name": item.get("name") or None,
            "quantity": normalize_qty(item.get("quantity")),
            "price": normalize_amount(item.get("price")),
        })
    return record


def gt_from_cord(gt_parse):
    """CORD ground truth -> our schema. Only the evaluator uses this."""
    menu = gt_parse.get("menu", [])
    if isinstance(menu, dict):
        menu = [menu]
    sub_total = gt_parse.get("sub_total")
    if not isinstance(sub_total, dict):
        sub_total = {}
    total = gt_parse.get("total")
    if not isinstance(total, dict):
        total = {}

    items = []
    for m in menu:
        if isinstance(m, dict):
            items.append({"name": m.get("nm"), "quantity": m.get("cnt"), "price": m.get("price")})

    return coerce_record({
        "line_items": items,
        "subtotal": sub_total.get("subtotal_price"),
        "tax": sub_total.get("tax_price"),
        "service": sub_total.get("service_price"),
        "discount": sub_total.get("discount_price"),
        "total": total.get("total_price"),
    })


def item_key(item):
    """An item is compared by (quantity, price). Missing quantity counts as 1."""
    quantity = item["quantity"]
    if quantity is None:
        quantity = 1
    return (quantity, item["price"])


def item_bag(record):
    """Line items as a multiset: the order does not matter, duplicates are kept."""
    return Counter(item_key(item) for item in record["line_items"])


# ---- validator (business rules)

def rule(status, reason, fields):
    return {"status": status, "reason": reason, "fields": fields}


def validate(rec):
    """Checks rules R1-R4 on the extracted data. Each rule is passed, failed or not_applicable."""
    results = {}

    prices = [item["price"] for item in rec["line_items"] if item["price"] is not None]
    if prices:
        items_sum = sum(prices)
    else:
        items_sum = None

    # missing tax, service or discount count as 0
    adjustments = (rec["tax"] or 0) + (rec["service"] or 0) - (rec["discount"] or 0)

    # R1: the total must be present
    if rec["total"] is not None:
        results["R1_total_present"] = rule("passed", "total present", ["total"])
    else:
        results["R1_total_present"] = rule("failed", "total missing", ["total"])

    # R2: sum of items = subtotal
    r2_fields = ["line_items", "subtotal"]
    if items_sum is None or rec["subtotal"] is None:
        results["R2_items_eq_subtotal"] = rule("not_applicable", "line items or subtotal missing", r2_fields)
    else:
        status = "passed" if items_sum == rec["subtotal"] else "failed"
        reason = f"items {items_sum} vs subtotal {rec['subtotal']}"
        results["R2_items_eq_subtotal"] = rule(status, reason, r2_fields)

    # R3: subtotal + tax + service - discount = total
    if rec["subtotal"] is None or rec["total"] is None:
        results["R3_subtotal_adj_eq_total"] = rule("not_applicable", "subtotal or total missing", AMOUNT_FIELDS)
    else:
        calculated = rec["subtotal"] + adjustments
        status = "passed" if calculated == rec["total"] else "failed"
        reason = (f"{rec['subtotal']} + adj {adjustments} = {calculated} vs total {rec['total']} "
                  f"(missing tax/service/discount treated as 0)")
        results["R3_subtotal_adj_eq_total"] = rule(status, reason, AMOUNT_FIELDS)

    # R4: only when there is no subtotal: items + adjustments = total
    r4_fields = ["line_items", "tax", "service", "discount", "total"]
    if rec["subtotal"] is not None:
        results["R4_items_adj_eq_total"] = rule("not_applicable", "subtotal present, R3 used instead",
                                                ["line_items", "total"])
    elif items_sum is None or rec["total"] is None:
        results["R4_items_adj_eq_total"] = rule("not_applicable", "line items or total missing",
                                                ["line_items", "total"])
    else:
        calculated = items_sum + adjustments
        status = "passed" if calculated == rec["total"] else "failed"
        reason = f"items {items_sum} + adj {adjustments} = {calculated} vs total {rec['total']}"
        results["R4_items_adj_eq_total"] = rule(status, reason, r4_fields)

    return results


def critical_rules_ok(rules):
    """OK only if no rule failed and at least one of R3/R4 really passed."""
    for r in rules.values():
        if r["status"] == "failed":
            return False
    return (rules["R3_subtotal_adj_eq_total"]["status"] == "passed"
            or rules["R4_items_adj_eq_total"]["status"] == "passed")


# ---- confidence signals and routing

def rule_signal(field, rules):
    """1 if every applicable rule that uses this field passed, otherwise 0."""
    relevant = [r for r in rules.values() if field in r["fields"] and r["status"] != "not_applicable"]
    if not relevant:
        return 0  # nothing could be checked, so there is no evidence
    for r in relevant:
        if r["status"] == "failed":
            return 0
    return 1


def field_signals(primary, text_rec, ocr_numbers, rules):
    """Signals for every field of the image reading (primary).
    cross    = the OCR-text reading gives the same value
    evidence = the value appears in the OCR text
    rule     = the rules that use this field passed"""
    signals = {}

    for field in AMOUNT_FIELDS:
        value = primary[field]
        if value is None and text_rec[field] is None:
            continue  # the field is not on the receipt in both readings
        signals[field] = {
            "value": value,
            "cross": int(value == text_rec[field]),
            "evidence": int(value is not None and value in ocr_numbers),
            "rule": rule_signal(field, rules),
        }

    text_items = item_bag(text_rec)
    for i, item in enumerate(primary["line_items"]):
        key = item_key(item)
        signals[f"item_{i}"] = {
            "value": key,
            "cross": int(text_items.get(key, 0) > 0),
            "evidence": int(item["price"] is not None and item["price"] in ocr_numbers),
            "rule": rule_signal("line_items", rules),
        }
    return signals


def confidence(signal, after=False):
    """Weighted sum of the signals. After a re-read the repeat signal is added."""
    if after and "repeat" in signal:
        weights = PROVISIONAL["weights_after"]
    else:
        weights = PROVISIONAL["weights_initial"]
    total = 0
    for name, weight in weights.items():
        total += weight * signal.get(name, 0)
    return round(total, 3)


def p_error(conf):
    for minimum, probability in PROVISIONAL["p_error_table"]:
        if conf >= minimum:
            return probability
    return PROVISIONAL["p_error_table"][-1][1]


def expected_error_cost(signals, after=False):
    """E = sum over fields of p(error) x cost of that field."""
    E = 0.0
    detail = {}
    for field, signal in signals.items():
        conf = confidence(signal, after)
        p = p_error(conf)
        if field.startswith("item_"):
            cost = PROVISIONAL["error_cost"]["item_price"]
        else:
            cost = PROVISIONAL["error_cost"][field]
        detail[field] = {"conf": conf, "p_error": p, "risk": round(p * cost, 4)}
        E += p * cost
    return round(E, 4), detail


def decide_s1(rules):
    if critical_rules_ok(rules):
        return "approve"
    return "review"


def decide_s2(rules, E):
    if critical_rules_ok(rules) and E < PROVISIONAL["review_cost"]:
        return "approve"
    return "review"


def uncertain_fields(detail):
    fields = set()
    for field, d in detail.items():
        if d["conf"] < PROVISIONAL["uncertain_threshold"]:
            if field.startswith("item_"):
                fields.add("line_items")
            else:
                fields.add(field)
    return sorted(fields)


# ---- evaluator (the only place where ground truth is used)

def evaluate(rec, gt):
    """A receipt is correct only if all amount fields and all (quantity, price) items match.
    Item names are not critical; their match share is reported separately."""
    fields_ok = {}
    for field in AMOUNT_FIELDS:
        fields_ok[field] = rec[field] == gt[field]
    items_ok = item_bag(rec) == item_bag(gt)

    gt_names = Counter((item["name"] or "").strip().lower() for item in gt["line_items"])
    rec_names = Counter((item["name"] or "").strip().lower() for item in rec["line_items"])
    matched = sum((gt_names & rec_names).values())
    name_share = matched / max(1, sum(gt_names.values()))

    return {"correct": all(fields_ok.values()) and items_ok,
            "fields": fields_ok,
            "line_items_ok": items_ok,
            "item_name_match_share": round(name_share, 3)}


# ---- extraction: prompts, Gemini client, OCR

SCHEMA_TEXT = ('{"line_items": [{"name": str, "quantity": int|null, "price": int|null}], '
               '"subtotal": int|null, "tax": int|null, "service": int|null, "discount": int|null, '
               '"total": int|null}')

PROMPT_BASE = ("You extract data from a receipt. Return ONLY JSON with this schema: " + SCHEMA_TEXT +
               ". 'price' is the line total of an item. Use null if a field is not printed on the receipt "
               "(never 0 for a missing field). Copy values exactly as printed; do not calculate or correct them.")

PROMPT_TEXT = PROMPT_BASE + "\nThe receipt text (from OCR) is:\n{ocr}"

PROMPT_IMAGE = PROMPT_BASE + "\nThe receipt is in the attached image."

PROMPT_REREAD = ("Look at the receipt image again and re-read ONLY these fields: {fields}. "
                 "Read what is printed; do not change values to make the arithmetic work. "
                 "Return ONLY JSON: {\"values\": {<field>: value|null, ...}, "
                 "\"quotes\": {<field>: \"exact text you read\"}}. "
                 "If line_items is requested, return the full list in this schema: "
                 "[{\"name\": str, \"quantity\": int|null, \"price\": int|null}], where price is the line total.")

PROMPT_VERSION = "v3"


class LLMClient:
    """Sends requests to Gemini and saves every answer in a cache folder,
    together with the model, settings, tokens, cost and time."""

    def __init__(self, api_key, model_id, price_in_per_1m, price_out_per_1m, cache_dir="cache",
                 thinking_level="low", timeout_s=120, min_interval_s=13, verbose=True):
        from google import genai
        from google.genai import types
        self.client = genai.Client(api_key=api_key,
                                   http_options=types.HttpOptions(timeout=timeout_s * 1000))
        self.model_id = model_id
        self.price_in = price_in_per_1m
        self.price_out = price_out_per_1m
        self.thinking_level = thinking_level
        self.min_interval_s = min_interval_s  # pause between calls because of the free-tier limit
        self.verbose = verbose
        self.last_call_time = 0.0
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def cache_path(self, kind, receipt_id, prompt):
        text = f"{self.model_id}|{PROMPT_VERSION}|{kind}|{receipt_id}|{prompt}"
        key = hashlib.sha1(text.encode()).hexdigest()[:16]
        return os.path.join(self.cache_dir, f"{receipt_id}_{kind}_{key}.json")

    def call(self, kind, receipt_id, prompt, image_png=None, max_technical_retries=5):
        """kind is 'text', 'image' or 'reread'. Technical retries (the API request failed)
        are counted separately from the selective re-read, which is its own call."""
        path = self.cache_path(kind, receipt_id, prompt)
        if os.path.exists(path):
            with open(path) as f:
                saved = json.load(f)
            saved["from_cache"] = True
            return saved

        from google.genai import types
        if image_png is None:
            contents = [prompt]
        else:
            contents = [types.Part.from_bytes(data=image_png, mime_type="image/png"), prompt]
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            max_output_tokens=8192,
            thinking_config=types.ThinkingConfig(thinking_level=self.thinking_level),
        )

        errors = []
        for attempt in range(max_technical_retries + 1):
            # wait so that we stay under the requests-per-minute limit
            wait = self.min_interval_s - (time.time() - self.last_call_time)
            if wait > 0:
                time.sleep(wait)
            self.last_call_time = time.time()

            start = time.time()
            try:
                response = self.client.models.generate_content(
                    model=self.model_id, contents=contents, config=config)
            except Exception as e:
                message = str(e)
                errors.append(f"attempt {attempt}: {type(e).__name__}: {message[:200]}")
                if self.verbose:
                    print(f"  {receipt_id} {kind}: attempt {attempt} failed: {message[:300]}", flush=True)
                if "429" in message or "RESOURCE_EXHAUSTED" in message:
                    time.sleep(60)  # too many requests: wait for the next minute
                elif "503" in message or "UNAVAILABLE" in message:
                    time.sleep(20)  # Google is busy
                else:
                    time.sleep(2 ** attempt)
                continue

            seconds = time.time() - start
            usage = response.usage_metadata
            tokens_in = usage.prompt_token_count or 0
            tokens_thinking = getattr(usage, "thoughts_token_count", 0) or 0
            tokens_out = (usage.candidates_token_count or 0) + tokens_thinking  # thinking is billed as output
            cost = tokens_in / 1e6 * self.price_in + tokens_out / 1e6 * self.price_out

            result = {"receipt_id": receipt_id, "kind": kind, "model_id": self.model_id,
                      "settings": {"thinking_level": self.thinking_level, "prompt_version": PROMPT_VERSION},
                      "raw": response.text,
                      "tokens_in": tokens_in, "tokens_out": tokens_out, "tokens_thinking": tokens_thinking,
                      "cost_usd": cost, "seconds": round(seconds, 2),
                      "technical_retries": attempt, "technical_errors": errors}
            with open(path, "w") as f:
                json.dump(result, f, indent=1)
            result["from_cache"] = False
            if self.verbose:
                print(f"  {receipt_id} {kind}: {seconds:.1f}s, tokens in/out/thinking "
                      f"{tokens_in}/{tokens_out}/{tokens_thinking}", flush=True)
            return result

        # every attempt failed
        return {"receipt_id": receipt_id, "kind": kind, "model_id": self.model_id, "raw": None,
                "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0, "seconds": 0,
                "technical_retries": max_technical_retries, "technical_errors": errors,
                "from_cache": False}


def parse_llm(raw):
    """Reads the JSON from the answer. Returns (data, error message)."""
    if raw is None:
        return None, "no response (technical failure)"
    match = re.search(r"\{.*\}", raw.strip(), re.S)
    text = match.group(0) if match else raw.strip()
    try:
        return json.loads(text), None
    except Exception as e:
        return None, f"JSON parse error: {e}"


def run_ocr(pil_image):
    """Tesseract on the full-size photo: grayscale, stronger contrast, at least 1600 px wide.
    Three page modes are tried and the longest result is kept."""
    import pytesseract
    from PIL import ImageOps
    img = ImageOps.autocontrast(ImageOps.grayscale(pil_image))
    if img.width < 1600:
        scale = 1600 / img.width
        img = img.resize((1600, int(img.height * scale)))

    best = ""
    for psm in (6, 4, 11):
        text = pytesseract.image_to_string(img, config=f"--psm {psm}")
        if len(text.strip()) > len(best.strip()):
            best = text
        if len(best.strip()) > 80:
            break
    return best


# ---- one receipt: initial extraction

def process_receipt(receipt_id, image_png, pil_image, llm, ocr_text=None):
    """OCR + two Gemini readings (from text and from image). S0-S3 all use this same result."""
    if ocr_text is None:
        ocr_text = run_ocr(pil_image)

    text_call = llm.call("text", receipt_id, PROMPT_TEXT.replace("{ocr}", ocr_text))
    image_call = llm.call("image", receipt_id, PROMPT_IMAGE, image_png=image_png)

    text_answer, text_error = parse_llm(text_call["raw"])
    image_answer, image_error = parse_llm(image_call["raw"])
    parse_errors = [e for e in (text_error, image_error) if e]

    return {"receipt_id": receipt_id,
            "ocr_text": ocr_text,
            "calls": [text_call, image_call],
            "parse_errors": parse_errors,
            "text_rec": coerce_record(text_answer),
            "image_rec": coerce_record(image_answer),  # the image reading is the primary one
            "ocr_numbers": sorted(numbers_in_text(ocr_text))}


# ---- re-extraction with guardrails

def apply_reread(primary, parsed, fields):
    """Puts the re-read values into a copy of the original.
    Guardrail G1: a re-read never replaces a value with null, and an item list
    with missing prices never replaces the original list. Rejected changes are logged."""
    new = copy.deepcopy(primary)
    values = {}
    if isinstance(parsed, dict):
        values = parsed.get("values") or {}
    rejected = []

    for field in fields:
        if field == "line_items" and isinstance(values.get("line_items"), list):
            items = coerce_record({"line_items": values["line_items"]})["line_items"]
            if items and all(item["price"] is not None for item in items):
                new["line_items"] = items
            else:
                rejected.append("line_items: incomplete list (missing prices)")
        elif field in AMOUNT_FIELDS and field in values:
            value = normalize_amount(values.get(field))
            if value is None and primary.get(field) is not None:
                rejected.append(f"{field}: null would replace {primary.get(field)}")
            else:
                new[field] = value
    return new, rejected


def after_signals(primary_before, new_rec, text_rec, ocr_numbers, quotes, rules_new):
    """Signals after the re-read. Cross agreement stays as it was;
    repeat agreement (re-read = original image reading) is a new, separate signal."""
    signals = field_signals(new_rec, text_rec, ocr_numbers, rules_new)
    items_before = item_bag(primary_before)

    for field, signal in signals.items():
        if field.startswith("item_"):
            signal["repeat"] = int(items_before.get(signal["value"], 0) > 0)
            continue
        signal["repeat"] = int(new_rec[field] == primary_before[field])
        quote = quotes.get(field)
        if quote is not None:
            quoted_number = normalize_amount(quote)
            signal["evidence"] = int(quoted_number is not None and quoted_number == new_rec[field]
                                     and quoted_number in ocr_numbers)
    return signals


# ---- routing S0-S3 for one receipt

def route_all(log, llm=None, image_png=None):
    """Applies S0, S1, S2 and S3 to the same initial extraction.
    Only S3 may make one extra re-read call."""
    primary = log["image_rec"]
    text_rec = log["text_rec"]
    ocr_numbers = set(log["ocr_numbers"])

    rules = validate(primary)
    signals = field_signals(primary, text_rec, ocr_numbers, rules)
    E, detail = expected_error_cost(signals)

    out = {
        "S0": {"decision": "approve", "final": primary, "extra_calls": 0},
        "S1": {"decision": decide_s1(rules), "final": primary, "extra_calls": 0},
        "S2": {"decision": decide_s2(rules, E), "final": primary, "extra_calls": 0, "E": E},
    }
    s3 = {"decision": out["S2"]["decision"], "final": primary, "extra_calls": 0, "E_before": E,
          "reextraction": None, "rules_initial": rules, "signals_initial": signals,
          "detail_initial": detail}
    out["S3"] = s3

    # S3 re-reads only if: S2 would send to review, some fields are uncertain,
    # no re-read happened yet, and the call is cheaper than the expected error cost
    fields = uncertain_fields(detail)
    if s3["decision"] != "review" or not fields or llm is None:
        return out
    if PROVISIONAL["max_reextraction_rounds"] < 1:
        return out
    if PROVISIONAL["reread_call_cost_estimate"] >= E:
        return out

    prompt = PROMPT_REREAD.replace("{fields}", ", ".join(fields))
    call = llm.call("reread", log["receipt_id"], prompt, image_png=image_png)
    parsed, parse_error = parse_llm(call["raw"])
    quotes = {}
    if isinstance(parsed, dict):
        quotes = parsed.get("quotes") or {}

    new, rejected = apply_reread(primary, parsed, fields)
    rules_after = validate(new)

    # Guardrail G2: if the rules passed before, the re-read may not break them
    if critical_rules_ok(rules) and not critical_rules_ok(rules_after):
        rejected.append("G2: re-read broke rules that passed before; original values kept")
        new = copy.deepcopy(primary)
        rules_after = rules

    signals_after = after_signals(primary, new, text_rec, ocr_numbers, quotes, rules_after)
    E_after, detail_after = expected_error_cost(signals_after, after=True)
    decision = decide_s2(rules_after, E_after)

    s3["decision"] = decision
    s3["final"] = new
    s3["extra_calls"] = 1
    s3["E_after"] = E_after
    s3["reextraction"] = {
        "trigger": f"E={E} >= review cost {PROVISIONAL['review_cost']} or critical rule not ok",
        "fields_requested": fields,
        "original_values": {f: primary.get(f) for f in fields},
        "new_values": {f: new.get(f) for f in fields},
        "confidence_before": {k: v["conf"] for k, v in detail.items()},
        "confidence_after": {k: v["conf"] for k, v in detail_after.items()},
        "critical_rules_before_ok": critical_rules_ok(rules),
        "critical_rules_after_ok": critical_rules_ok(rules_after),
        "rules_after": rules_after,
        "parse_error": parse_error,
        "guardrail_rejections": rejected,
        "cost_usd": call["cost_usd"],
        "from_cache": call.get("from_cache"),
    }
    return out


# ---- results table

def summarize(results, gts, call_costs_usd, usd_to_eur=0.92):
    """One row per strategy. gts = ground truth, used only here."""
    n = len(results)
    rows = {}
    for S in ["S0", "S1", "S2", "S3"]:
        approved = 0
        wrong_approved = 0
        correct_to_review = 0
        extra_calls = 0
        for result, gt in zip(results, gts):
            d = result[S]
            is_correct = evaluate(d["final"], gt)["correct"]
            extra_calls += d.get("extra_calls", 0)
            if d["decision"] == "approve":
                approved += 1
                if not is_correct:
                    wrong_approved += 1
            elif is_correct:
                correct_to_review += 1

        api_cost = call_costs_usd.get("initial", 0.0)
        if S == "S3":
            api_cost += call_costs_usd.get("reread", 0.0)

        rows[S] = {
            "receipts": n,
            "STP_rate": round(approved / n, 3) if n else None,
            "silent_error_rate (wrong approvals / all)": round(wrong_approved / n, 3) if n else None,
            "approval_error_rate (wrong approvals / approved)": round(wrong_approved / approved, 3) if approved else None,
            "correct_receipts_sent_to_review": correct_to_review,
            "API_calls": 2 * n + extra_calls,
            "measured_API_cost_EUR": round(api_cost * usd_to_eur, 4),
            "assumed_review_cost_EUR": round((n - approved) * PROVISIONAL["review_cost"], 2),
            "assumed_error_cost_EUR (total-field cost per wrong approval)":
                round(wrong_approved * PROVISIONAL["error_cost"]["total"], 2),
        }
    return rows
