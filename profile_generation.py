import asyncio
import time

from config import get_model_name
from generator import LLMGenerator


PROMPT = (
    "Explain how photosynthesis works in simple terms. "
    "Keep the explanation concise."
)


async def profile_main_response(generator):
    start = time.perf_counter()

    await generator.generate_main_response(PROMPT)

    elapsed = time.perf_counter() - start

    print(f"Main response: {elapsed:.2f} sec")


async def profile_samples(generator, num_samples):
    start = time.perf_counter()

    await generator.generate_stochastic_samples(
        PROMPT,
        num_samples,
    )

    elapsed = time.perf_counter() - start

    print(
        f"{num_samples} samples: "
        f"{elapsed:.2f} sec"
    )


async def profile_complete_pipeline(generator, num_samples):
    start = time.perf_counter()

    await generator.generate_all(
        prompt=PROMPT,
        num_samples=num_samples,
    )

    elapsed = time.perf_counter() - start

    print(
        f"Complete pipeline ({num_samples} samples): "
        f"{elapsed:.2f} sec"
    )


async def main():
    generator = LLMGenerator(
        model_name=get_model_name(),
        max_concurrency=5,
    )

    print("=== SelfCheckGPT Generation Performance ===")
    print()

    await profile_main_response(generator)

    print()

    for num_samples in [4, 10, 20]:
        await profile_samples(
            generator,
            num_samples,
        )

    print()

    for num_samples in [4, 10, 20]:
        await profile_complete_pipeline(
            generator,
            num_samples,
        )


if __name__ == "__main__":
    asyncio.run(main())