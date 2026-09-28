from eval_ngram import score_sentences as score_unigram
from eval_prompt import score_sentences as score_prompt
from eval_nli import NLIScorer


class SelfCheckEvaluator:
    def __init__(self):
        self._nli_scorer = None

    def _get_nli_scorer(self):
        if self._nli_scorer is None:
            self._nli_scorer = NLIScorer()
        return self._nli_scorer

    def _normalize_scores(self, scores):
        """Day 5 Requirement: Bound all output scores strictly between 0.0 and 1.0"""
        if not scores:
            return []
        max_val = max(scores)
        if max_val > 1.0:
            return [round(s / max_val, 4) for s in scores]
        return [round(max(0.0, min(1.0, float(s))), 4) for s in scores]

    def evaluate(self, sentences, samples, method="unigram", evaluator_fn=None):
        if method == "unigram":
            raw_scores = score_unigram(sentences, samples)
        elif method == "prompt":
            if evaluator_fn is None:
                raise ValueError("evaluator_fn is required for prompt scoring.")
            raw_scores = score_prompt(sentences, samples, evaluator_fn)
        elif method == "nli":
            nli = self._get_nli_scorer()
            raw_scores = [nli.score_sentence(s, samples) for s in sentences]
        else:
            raise ValueError(f"Unknown method: {method}. Use 'unigram', 'prompt', or 'nli'.")

        scores = self._normalize_scores(raw_scores)
        passage_score = round(sum(scores) / len(scores), 4) if scores else 0.0

        return {
            "sentence_scores": scores,
            "passage_score": passage_score
        }


if __name__ == "__main__":
    evaluator = SelfCheckEvaluator()

    samples = [
        "The Eiffel Tower is located in Paris.",
        "Paris is the capital of France and home to the Eiffel Tower."
    ]
    sentences = [
        "The Eiffel Tower is located in Paris.",
        "The Eiffel Tower is in London."
    ]

    print("--- 1. Testing Unigram Scorer ---")
    print(evaluator.evaluate(sentences, samples, method="unigram"))

    print("\n--- 2. Testing Prompt Scorer ---")
    def mock_llm(prompt):
        return "Yes" if "Paris" in prompt else "No"
    print(evaluator.evaluate(sentences, samples, method="prompt", evaluator_fn=mock_llm))

    print("\n--- 3. Testing NLI Scorer ---")
    print(evaluator.evaluate(sentences, samples, method="nli"))
