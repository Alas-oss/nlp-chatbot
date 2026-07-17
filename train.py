import sys
sys.path.insert(0, "src")

from intent_classifier import train

train("data/intents.json", "intent_model.pkl")