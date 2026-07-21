import sys
sys.path.insert(0, "src")
from archive.intent_classifier import predict

def test_greeting_matches():
    response, confidence = predict("hello there", "intent_model.pkl")
    assert response is not None
    assert confidence > 0.4

def test_gibberish_falls_back():
    response, confidence = predict("asdkfj qwoeiru zzz", "intent_model.pkl")
    assert response is None