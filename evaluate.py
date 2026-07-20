import sys
sys.path.insert(0, "src")
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from intent_classifier import train
from preprocessing import preprocess
import pickle

with open("data/intents.json") as f:
    data = json.load(f)["intents"]


X_raw, y = [], []
for intent in data:
    for pattern in intent["patterns"]:
        X_raw.append(pattern)
        y.append(intent["tag"])

X_train, X_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42, stratify=y)

with open("intent_model.pkl", "rb") as f:
    bundle = pickle.load(f)

X_test_vec = bundle["vectorizer"].transform([preprocess(t) for t in X_test])
y_pred = bundle["classifier"].predict(X_test_vec)

print(classification_report(y_test, y_pred))
print(f"Confusion matrix:\n {confusion_matrix(y_test, y_pred)}")