from typing import List, Dict


class SelfCheckEvaluator:
    """
    Base interface for SelfCheckGPT hallucination scoring.

    Accepts:
        sentences: sentences from the main response R
        samples: stochastic responses S^1 ... S^N

    Returns:
        Sentence-level hallucination scores in [0.0, 1.0]
    """

    def __init__(
        self,
        sentences: List[str],
        samples: List[str]
    ):
        self.sentences = sentences
        self.samples = samples

    def score_prompt(self) -> List[float]:
        raise NotImplementedError

    def score_nli(self) -> List[float]:
        raise NotImplementedError

    def score_unigram(self) -> List[float]:
        raise NotImplementedError

    def score_all(self) -> Dict[str, List[float]]:
        return {
            "prompt": self.score_prompt(),
            "nli": self.score_nli(),
            "unigram": self.score_unigram(),
        }