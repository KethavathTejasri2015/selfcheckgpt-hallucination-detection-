import asyncio
import logging

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import get_api_key
from preprocessor import preprocess_response


logger = logging.getLogger(__name__)


class LLMGenerator:
    """
    Asynchronous LLM generation engine for SelfCheckGPT.

    Handles deterministic main-response generation, stochastic
    sample generation, concurrency control, and transient API retries.
    """

    def __init__(
        self,
        model_name: str,
        max_concurrency: int = 5,
    ) -> None:
        """
        Initialize the LLM generation engine.

        Args:
            model_name: Name of the language model to use.
            max_concurrency: Maximum number of stochastic API requests
                allowed to run concurrently.
        """
        self.model_name = model_name

        self.client = AsyncOpenAI(
            api_key=get_api_key(),
            base_url="https://api.groq.com/openai/v1",
        )

        self.semaphore = asyncio.Semaphore(max_concurrency)

        logger.info(
            "LLMGenerator initialized with model=%s, "
            "max_concurrency=%d",
            self.model_name,
            max_concurrency,
        )

    @retry(
        retry=retry_if_exception_type(
            (
                RateLimitError,
                InternalServerError,
                APIConnectionError,
                APITimeoutError,
            )
        ),
        stop=stop_after_attempt(5),
        wait=wait_exponential(
            multiplier=1,
            min=1,
            max=20,
        ),
    )
    async def generate_main_response(
        self,
        prompt: str,
    ) -> str:
        """
        Generate the primary deterministic response.

        The main response uses temperature 0.0 to minimize
        randomness and provide a consistent response.

        Args:
            prompt: User input used for generation.

        Returns:
            The generated primary response as a string.

        Raises:
            RuntimeError: If the model returns an empty response.
        """
        logger.info("Generating main response")

        response = await self.client.chat.completions.create(
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
                "Model returned an empty response."
            )

        logger.info("Main response generation completed")

        return content.strip()

    @retry(
        retry=retry_if_exception_type(
            (
                RateLimitError,
                InternalServerError,
                APIConnectionError,
                APITimeoutError,
            )
        ),
        stop=stop_after_attempt(5),
        wait=wait_exponential(
            multiplier=1,
            min=1,
            max=20,
        ),
    )
    async def generate_sample(
        self,
        prompt: str,
    ) -> str:
        """
        Generate one stochastic response.

        A temperature of 1.0 is used to introduce variation between
        the generated samples for SelfCheckGPT evaluation.

        Args:
            prompt: User input used for generation.

        Returns:
            A single generated stochastic response.

        Raises:
            RuntimeError: If the model returns an empty response.
        """
        async with self.semaphore:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=1.0,
            )

        content = response.choices[0].message.content

        if not content:
            raise RuntimeError(
                "Model returned an empty sample."
            )

        return content.strip()

    async def generate_stochastic_samples(
        self,
        prompt: str,
        num_samples: int,
    ) -> list[str]:
        """
        Generate multiple stochastic responses for the same prompt.

        The requests are executed concurrently while the semaphore
        limits the maximum number of simultaneous API calls.

        Args:
            prompt: User input used for generation.
            num_samples: Number of stochastic responses to generate.
                Must be between 4 and 20 inclusive.

        Returns:
            A list containing the generated stochastic responses.

        Raises:
            ValueError: If num_samples is outside the range 4–20.
        """
        if not 4 <= num_samples <= 20:
            raise ValueError(
                "num_samples must be between 4 and 20"
            )

        logger.info(
            "Generating %d stochastic samples",
            num_samples,
        )

        tasks = [
            self.generate_sample(prompt)
            for _ in range(num_samples)
        ]

        samples = await asyncio.gather(*tasks)

        logger.info(
            "Stochastic sample generation completed"
        )

        return list(samples)

    async def generate_all(
        self,
        prompt: str,
        num_samples: int,
    ) -> dict:
        """
        Run the complete SelfCheckGPT generation pipeline.

        The pipeline:
        1. Generates the primary response.
        2. Splits the primary response into sentences.
        3. Generates stochastic samples for comparison.

        Args:
            prompt: User input used for generation.
            num_samples: Number of stochastic responses to generate.
                Must be between 4 and 20 inclusive.

        Returns:
            A dictionary containing:
                - response: Primary generated response.
                - sentences: Sentence-level segmentation of the response.
                - samples: Stochastic generated responses.
        """
        logger.info("Starting complete generation pipeline")

        main_response = await self.generate_main_response(
            prompt
        )

        sentences = preprocess_response(
            main_response
        )

        logger.info(
            "Sentence segmentation completed: %d sentences",
            len(sentences),
        )

        samples = await self.generate_stochastic_samples(
            prompt,
            num_samples,
        )

        logger.info(
            "Complete generation pipeline finished"
        )

        return {
            "response": main_response,
            "sentences": sentences,
            "samples": samples,
        }