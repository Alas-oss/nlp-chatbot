import sys
sys.path.insert(0, "src")

from archive.intent_classifier import train

train("data/intents.json", "intent_model.pkl")