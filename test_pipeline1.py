import json
from receipt_pipeline import *

def rec(items, subtotal=None, tax=None, total=None, service=None, discount=None):
    return coerce_record({"line_items":[{"name":n,"quantity":q,"price":p} for n,q,p in items],
                          "subtotal":subtotal,"tax":tax,"service":service,"discount":discount,"total":total})
st = lambda r,k: validate(r)[k]["status"]

# normalisation
assert normalize_amount("Rp 12.500")==12500 and normalize_amount("12,500")==12500 and normalize_amount(None) is None
assert normalize_amount("")is None and normalize_amount("-5,000")==-5000

gt = rec([("Nasi goreng",1,35000),("Es teh",1,25000)],60000,6000,66000)
# 1 correct extraction
r = rec([("Es teh",1,25000),("Nasi goreng",1,35000)],60000,6000,66000)
assert st(r,"R2_items_eq_subtotal")=="passed" and st(r,"R3_subtotal_adj_eq_total")=="passed" and critical_rules_ok(validate(r))
assert evaluate(r,gt)["correct"]            # order of items does not matter
# 2 incorrect total
r = rec([("Nasi goreng",1,35000),("Es teh",1,25000)],60000,6000,76000)
assert st(r,"R3_subtotal_adj_eq_total")=="failed" and not critical_rules_ok(validate(r)) and not evaluate(r,gt)["correct"]
# 3a missing tax (treated as 0 -> rule fails here because receipt total includes tax)
r = rec([("Nasi goreng",1,35000),("Es teh",1,25000)],60000,None,66000)
assert st(r,"R3_subtotal_adj_eq_total")=="failed"
# 3b missing subtotal -> R2,R3 not applicable, R4 used
r = rec([("Coffee",1,10000),("Cake",1,15000)],None,None,25000)
v=validate(r); assert v["R2_items_eq_subtotal"]["status"]=="not_applicable" and v["R3_subtotal_adj_eq_total"]["status"]=="not_applicable" and v["R4_items_adj_eq_total"]["status"]=="passed"
# 3c missing total -> R1 fails, cannot approve
r = rec([("Coffee",1,10000)],10000,None,None); assert st(r,"R1_total_present")=="failed" and not critical_rules_ok(validate(r))
# 3d nothing checkable -> not approved
r = rec([],None,None,25000); assert not critical_rules_ok(validate(r))
# 4 inconsistent line items
r = rec([("Nasi goreng",1,35000),("Es teh",1,35000)],60000,6000,66000)
assert st(r,"R2_items_eq_subtotal")=="failed" and not critical_rules_ok(validate(r))
# 5 duplicate line items preserved
gt2 = rec([("Coffee",1,10000),("Coffee",1,10000)],20000,None,20000)
r1 = rec([("Coffee",1,10000)],20000,None,20000)
assert not evaluate(r1,gt2)["line_items_ok"]            # missing duplicate detected
assert st(r1,"R2_items_eq_subtotal")=="failed"          # and the validator notices it too
assert evaluate(rec([("Coffee",1,10000),("Coffee",1,10000)],20000,None,20000),gt2)["correct"]

# ---------- routing with a fake LLM (no API) : Example A and Example B from the design
class FakeLLM:
    def __init__(self, answer): self.answer=answer; self.calls=0
    def call(self, kind, rid, prompt, image_png=None):
        self.calls+=1; return {"raw": json.dumps(self.answer), "cost_usd": 0.0001, "from_cache": False}
ocrA = "NASI GORENG 35,000\nES TEH 25,000\nSUBTOTAL 60,000\nTAX 6,000\nTOTAL 66,000\nCASH 70,000"
logA = {"receipt_id":"A","ocr_numbers":sorted(numbers_in_text(ocrA)),
        "image_rec": rec([("Nasi goreng",1,35000),("Es teh",1,25000)],60000,6000,66000),
        "text_rec":  rec([("Nasi goreng",1,35000),("Es teh",1,25000)],60000,6000,70000)}
outA = route_all(logA, FakeLLM({"values":{"total":"66,000"},"quotes":{"total":"TOTAL 66,000"}}))
print("Example A:", {S:outA[S]["decision"] for S in outA}, "E_before", outA["S3"]["E_before"], "E_after", outA["S3"].get("E_after"))
print("   total conf before/after:", outA["S3"]["reextraction"]["confidence_before"]["total"], outA["S3"]["reextraction"]["confidence_after"]["total"])
ocrB = "SUBTOTAL 60,000\nTAX 6,000\nTOTAL 76,000\nNASI 35,000\nES TEH 25,000"
logB = {"receipt_id":"B","ocr_numbers":sorted(numbers_in_text(ocrB)),
        "image_rec": rec([("Nasi",1,35000),("Es teh",1,25000)],60000,6000,76000),
        "text_rec":  rec([("Nasi",1,35000),("Es teh",1,25000)],60000,6000,76000)}
outB = route_all(logB, FakeLLM({"values":{"subtotal":"60,000","tax":"6,000","total":"76,000"},"quotes":{"total":"TOTAL 76,000"}}))
print("Example B:", {S:outB[S]["decision"] for S in outB})
assert outB["S3"]["decision"]=="review" and outB["S1"]["decision"]=="review" and outB["S0"]["decision"]=="approve"
# summary table with evaluator
gtA = rec([("Nasi goreng",1,35000),("Es teh",1,25000)],60000,6000,66000); gtB = rec([("Nasi",1,35000),("Es teh",1,25000)],60000,6000,66000)
print(json.dumps(summarize([outA,outB],[gtA,gtB],{"initial":0.0004,"reread":0.0002}),indent=1))
# guardrails on re-extraction (found in development run 1)
orig = rec([("Baso tahu",1,43181),("Es jeruk",1,13000)],56181,5618,61799)
new, rej = apply_reread(orig, {"values":{"line_items":[{"name":None,"quantity":1,"price":None}],"total":None}}, ["line_items","total"])
assert new["line_items"]==orig["line_items"] and new["total"]==61799 and len(rej)==2   # G1
logG = {"receipt_id":"G","ocr_numbers":[], "image_rec": orig, "text_rec": empty_record()}
outG = route_all(logG, FakeLLM({"values":{"total":"62,000"},"quotes":{}}))
assert outG["S3"]["final"]["total"]==61799 and any("G2" in r for r in outG["S3"]["reextraction"]["guardrail_rejections"])  # G2
print("ALL TESTS PASSED")
