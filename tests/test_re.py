import spacy
import brand_detector as bd

def test_blank_model_smoke():
    nlp = spacy.blank("en")

def test_find_brands():
    text = "Hello T-Mobile for a limited time only. aDunkin! Dunkin!"
    ans1 = bd.utils.find_brands("T mobile", text, ignore_case=True, dehyphenate=False)
    assert ans1 == []
    ans2 = bd.utils.find_brands("T mobile", text, ignore_case=True, dehyphenate=True)
    assert ans2 == [(6, 14)]
    ans3 = bd.utils.find_brands("Dunkin'", text, ignore_case=False, dehyphenate=True)
    assert ans3 == [(31, 37)]