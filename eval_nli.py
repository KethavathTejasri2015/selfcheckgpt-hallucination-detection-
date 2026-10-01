import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

MODEL_NAME = "cross-encoder/nli-deberta-v3-large"

class NLIScorer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        self.model.eval()

    def contradiction_probability(self, premise, hypothesis):
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits[0]

        label_map = {
            label.lower(): index
            for index, label in self.model.config.id2label.items()
        }

        contradiction_index = label_map["contradiction"]
        entailment_index = label_map["entailment"]

        contradiction_logit = logits[contradiction_index]
        entailment_logit = logits[entailment_index]

        denominator = (
            torch.exp(entailment_logit)
            + torch.exp(contradiction_logit)
        )

        probability = (
            torch.exp(contradiction_logit)
            / denominator
        )

        return probability.item()

    def score_sentence(self, sentence, samples):
        scores = []
        for sample in samples:
            score = self.contradiction_probability(
                premise=sample,
                hypothesis=sentence
            )
            scores.append(score)

        if not scores:
            return 0.0

        return sum(scores) / len(scores)

if __name__ == "__main__":
    scorer = NLIScorer()
    samples = [
        "The Eiffel Tower is located in Paris.",
        "Paris is home to the Eiffel Tower."
    ]
    sentence = "The Eiffel Tower is located in London."
    
    score = scorer.score_sentence(sentence, samples)
    print("NLI score:", score)
