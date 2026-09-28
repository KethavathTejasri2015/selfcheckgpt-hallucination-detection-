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

    def __init__(
        self,
        model_name: str,
        max_concurrency: int = 5,
    ):
        self.model_name = model_name

        self.client = AsyncOpenAI(
            api_key=get_api_key()
        )

        self.semaphore = asyncio.Semaphore(
            max_concurrency
        )

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
        Generate the main deterministic response.
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
        Generate one stochastic sample.
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
        Generate 4 to 20 stochastic samples concurrently.
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
        Run the complete generation pipeline.
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

        logger.info("Complete generation pipeline finished")

        return {
            "response": main_response,
            "sentences": sentences,
            "samples": samples,
        }