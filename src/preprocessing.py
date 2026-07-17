import re
import spacy

nlp = spacy.load("en_core_web_sm")


def preprocess(text: str) -> str:
    text = re.sub(r"[^\w\s]", "", text.lower())
    doc = nlp(text)
    tokens = [tok.lemma_ for tok in doc if not tok.is_stop and tok.text.strip()]
    return " ".join(tokens)