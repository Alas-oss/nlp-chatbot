import json
import pickle
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from preprocessing import preprocess


def train(intents_path: str, model_out: str):
    with open(intents_path, encoding="utf-8") as f:
        data = json.load(f)["intents"]

    X_raw, y = [], []
    for intent in data:
        for pattern in intent["patterns"]:
            X_raw.append(preprocess(pattern))
            y.append(intent["tag"])

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X_raw)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)

    with open(model_out, "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "classifier": clf, "intents": data}, f)

    print(f"Trained on {len(X_raw)} examples across {len(set(y))} intents.")


def predict(text: str, model_path: str, confidence_threshold: float = 0.4):
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    X = bundle["vectorizer"].transform([preprocess(text)])
    probs = bundle["classifier"].predict_proba(X)[0]
    best_idx = probs.argmax()
    confidence = probs[best_idx]
    tag = bundle["classifier"].classes_[best_idx]

    if confidence < confidence_threshold:
        return None, confidence

    for intent in bundle["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"]), confidence
    return None, confidence