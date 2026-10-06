from nltk.tokenize import sent_tokenize


def preprocess_response(text: str) -> list[str]:
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    text = text.strip()

    if not text:
        return []

    sentences = sent_tokenize(text)

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]