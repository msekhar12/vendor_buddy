from civsa.identity import (
    detect_doc_type,
    find_ids,
    gstin_check_char,
    name_matches,
    repair_gstin,
    repair_pan,
    validate_gstin,
    validate_pan,
)
from civsa.review import identity_reasons

G = "33AABCC4821K1ZW"          # fictional, checksum-valid


def test_gstin_checksum():
    assert gstin_check_char(G[:14]) == G[14]
    assert validate_gstin(G) == (True, "ok")
    assert validate_gstin(G[:14] + "X")[1] == "bad_checksum"
    assert validate_gstin("99AABCC4821K1Z" + gstin_check_char("99AABCC4821K1Z"))[1] == "bad_state_code"


def test_pan_validation():
    assert validate_pan("AABCC4821K")[0]
    assert validate_pan("AABQC4821K")[1] == "bad_holder_type"
    assert validate_pan("AABC4821K")[1] == "bad_format"


def test_ocr_repairs_by_position():
    assert repair_pan("AA8CC4B21K") == "AABCC4821K"        # B<->8 both ways
    assert repair_gstin("33AABCC4821KIZW") == G             # I -> 1 in entity slot
    assert repair_gstin("33AA8CC4821KI2W") == G             # 8->B, I->1, 2->Z


def test_words_never_become_ids():
    assert repair_pan("TECHNOLOGY") is None
    ids = find_ids("Indian Institute of Information Technology, GSTIN: " + G)
    assert ids == {"gstin": [G], "pan": []}                 # embedded PAN not repeated


def test_find_ids_with_spaces():
    assert find_ids("GSTIN 33 AABCC 4821K 1ZW")["gstin"] == [G]


def test_doc_type_detection():
    assert detect_doc_type("Permanent Account Number: AABCC4821K") == "pan_card"
    assert detect_doc_type("Goods and Services Tax Registration Certificate") == "gst_certificate"
    assert detect_doc_type("QUOTATION ... Unit Price ... Grand Total ... Payment Terms") == "quote"
    assert detect_doc_type("Masala Dosa Rs. 80") == "other"


def test_name_match_ignores_suffixes_typos_and_order():
    assert name_matches("RAJSHREE FASTENERS PRIVATE LIMITED", "Rajshree Fastners")
    assert name_matches("COROMANDEL SOLVENTS PRIVATE LIMITED", "Coromandel Solvents Pvt Ltd")
    assert name_matches("NIRMALA CHEMICALS PVT LTD", "Nirmala Chem")         # abbreviation
    assert name_matches("Associated Scientific Co.", "Associated Scientific Company")
    assert name_matches("SOLVENTS COROMANDEL LTD", "Coromandel Solvents")     # order


def test_name_match_rejects_extra_or_different_words():
    assert not name_matches("KONKAN MARINE EXPORTS PRIVATE LIMITED", "Konkan Chemicals Pvt Ltd")
    assert not name_matches("KONKAN MARINE EXPORTS PRIVATE LIMITED", "Konkan")   # was a prefix match
    assert not name_matches("Konkan", "Konkan Marine Exports")
    assert not name_matches("PINK CITY TRADERS", "Pink City Fasteners")
    assert not name_matches("AB", "ABC Labs")
    assert not name_matches("", "Konkan")


# ---- vendor-level rules ------------------------------------------------

def _doc(id_type, value, src):
    return {"id_type": id_type, "id_value": value, "source_doc": src}


def test_second_pan_document_flags():
    a, b = _doc("pan", "AAFCP5562Q", "a"), _doc("pan", "AAFCP5526Q", "b")
    assert "multiple_pan_docs" in identity_reasons(a, [b], [])
    assert "multiple_pan_docs" in identity_reasons(b, [a], [])


def test_one_pan_plus_one_gst_is_normal():
    pan = _doc("pan", "AABCC4821K", "p")
    gst = _doc("gstin", G, "g")
    assert identity_reasons(pan, [gst], []) == set()
    assert identity_reasons(gst, [pan], []) == set()


def test_pan_gstin_mismatch():
    pan = _doc("pan", "AAKCN6681B", "p")
    gst = _doc("gstin", G, "g")
    assert "pan_gstin_mismatch" in identity_reasons(pan, [gst], [])


def test_conflict_with_quote_uses_embedded_pan():
    pan = _doc("pan", "AAKCN6681B", "p")
    quote = {"gstin": "24AAKCN6618B1ZX", "pan": None, "source_doc": "q"}
    assert identity_reasons(pan, [], [quote]) == {"conflicts_with_quote"}
    ok_quote = {"gstin": "24AAKCN6681B1ZX", "pan": None, "source_doc": "q"}
    assert identity_reasons(pan, [], [ok_quote]) == set()


def test_multi_state_gstins_still_flagged():
    kl = _doc("gstin", "32AAJCM9034E1ZM", "kl")
    ka = _doc("gstin", "29AAJCM9034E1Z9", "ka")
    assert identity_reasons(kl, [ka], []) == {"multiple_gst_docs"}
