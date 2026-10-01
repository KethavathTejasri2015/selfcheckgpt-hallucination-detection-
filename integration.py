from generator import LLMGenerator
from scorer import SelfCheckEvaluator


class SelfCheckPipeline:

    def __init__(
        self,
        model_name: str,
        max_concurrency: int = 5,
    ):
        self.generator = LLMGenerator(
            model_name=model_name,
            max_concurrency=max_concurrency,
        )

        self.evaluator = SelfCheckEvaluator()

    async def generate_and_score(
        self,
        prompt: str,
        num_samples: int = 10,
        method: str = "unigram",
    ) -> dict:
        """
        Generate a response and evaluate its sentences
        against the stochastic samples.
        """

        generation_result = await self.generator.generate_all(
            prompt=prompt,
            num_samples=num_samples,
        )

        sentences = generation_result["sentences"]
        samples = generation_result["samples"]

        scores = self.evaluator.evaluate(
            sentences=sentences,
            samples=samples,
            method=method,
        )

        return {
            "response": generation_result["response"],
            "sentences": sentences,
            "samples": samples,
            "sentence_scores": scores["sentence_scores"],
            "passage_score": scores["passage_score"],
        }