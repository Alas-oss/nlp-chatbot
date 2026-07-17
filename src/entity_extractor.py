import spacy

nlp = spacy.load("en_core_web_sm")


def extract_entities(text: str) -> list[dict]:
    doc = nlp(text)
    return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]