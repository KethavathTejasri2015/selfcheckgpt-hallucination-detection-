import asyncio

from openai import OpenAI

from config import get_api_key
from generator import LLMGenerator
from scorer import SelfCheckEvaluator


class SelfCheckPipeline:
    """
    Complete SelfCheckGPT pipeline.

    Responsible for:
    1. Generating the main response.
    2. Generating stochastic samples.
    3. Splitting the main response into sentences.
    4. Passing sentences and samples to Person 2's evaluator.
    5. Supporting Unigram, Prompt, and NLI scoring.
    """

    def __init__(
        self,
        model_name: str,
        max_concurrency: int = 5,
    ):
        self.model_name = model_name

        # Person 1 generation engine
        self.generator = LLMGenerator(
            model_name=model_name,
            max_concurrency=max_concurrency,
        )

        # Person 2 scoring engine
        self.evaluator = SelfCheckEvaluator()

        # Synchronous client used only for Prompt-based evaluation.
        #
        # Person 2's eval_prompt.py expects a normal synchronous
        # evaluator(prompt) -> response callback.
        self.prompt_client = OpenAI(
            api_key=get_api_key()
        )

    def _prompt_evaluator(self, prompt: str) -> str:
        """
        Send a scoring prompt to the LLM and return its response.

        This callback matches the interface required by
        Person 2's eval_prompt.py.
        """

        response = self.prompt_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.0,
        )

        content = response.choices[0].message.content

        if not content:
            raise RuntimeError(
                "Prompt evaluator returned an empty response."
            )

        return content.strip()

    async def generate_and_score(
        self,
        prompt: str,
        num_samples: int = 10,
        method: str = "unigram",
    ) -> dict:
        """
        Generate a response and evaluate its sentences
        against stochastic samples.

        Supported methods:
            - unigram
            - prompt
            - nli
        """

        if method not in {"unigram", "prompt", "nli"}:
            raise ValueError(
                "method must be 'unigram', 'prompt', or 'nli'"
            )

        # --------------------------------------------------
        # STEP 1: Person 1 generation
        # --------------------------------------------------

        generation_result = await self.generator.generate_all(
            prompt=prompt,
            num_samples=num_samples,
        )

        sentences = generation_result["sentences"]
        samples = generation_result["samples"]

        # --------------------------------------------------
        # STEP 2: Person 2 scoring
        # --------------------------------------------------

        if method == "prompt":
            # Person 2's prompt scorer is synchronous.
            #
            # Run the synchronous evaluator in a worker thread
            # so it does not block the async generation pipeline.
            scores = await asyncio.to_thread(
                self.evaluator.evaluate,
                sentences,
                samples,
                method,
                self._prompt_evaluator,
            )
        else:
            scores = self.evaluator.evaluate(
                sentences=sentences,
                samples=samples,
                method=method,
            )

        # --------------------------------------------------
        # STEP 3: Combined result
        # --------------------------------------------------

        return {
            "response": generation_result["response"],
            "sentences": sentences,
            "samples": samples,
            "sentence_scores": scores["sentence_scores"],
            "passage_score": scores["passage_score"],
        }