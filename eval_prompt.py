def build_prompt(sentence, sample):
    return f"""Context: {sample}

Sentence: {sentence}

Is the sentence supported by the context above?

Answer Yes or No:
"""

def parse_response(response):
    text = response.strip().lower()

    if text.startswith("yes"):
        return 0.0

    if text.startswith("no"):
        return 1.0

    return 0.5

def score_sentence(sentence, samples, evaluator):
    scores = []

    for sample in samples:
        prompt = build_prompt(sentence, sample)
        response = evaluator(prompt)
        score = parse_response(response)
        scores.append(score)

    if not scores:
        return 0.5

    return sum(scores) / len(scores)

if __name__ == "__main__":
    def mock_evaluator(prompt):
        print("\nPROMPT:")
        print(prompt)
        return "Yes"

    samples = [
        "The cat is sitting on the mat.",
        "A cat is sitting on a mat."
    ]
    sentence = "The cat is sitting on the mat."
    
    score = score_sentence(sentence, samples, mock_evaluator)
    print("Score:", score)
def score_sentences(sentences, samples, evaluator):
    return [
        score_sentence(sentence, samples, evaluator)
        for sentence in sentences
    ]
