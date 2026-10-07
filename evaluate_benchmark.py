import asyncio
import json
import time
from pathlib import Path
from statistics import mean

from openai import OpenAI

from config import get_api_key, get_model_name
from generator import LLMGenerator
from preprocessor import preprocess_response
from scorer import SelfCheckEvaluator


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

PROMPTS_FILE = BASE_DIR / "data" / "test_prompts.json"
RESULTS_FILE = BASE_DIR / "benchmark_results.json"

NUM_SAMPLES = 4

MAX_RETRIES = 5
RETRY_WAIT_SECONDS = 20


# ============================================================
# API SETUP
# ============================================================

API_KEY = get_api_key()
MODEL_NAME = get_model_name()

if not API_KEY:
    raise RuntimeError(
        "API key not found. Please configure the API key."
    )

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.groq.com/openai/v1"
)


# ============================================================
# LOAD PROMPTS
# ============================================================

def load_test_prompts():
    with open(PROMPTS_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            "test_prompts.json must contain a list."
        )

    return data


# ============================================================
# PROMPT SCORER
# ============================================================

def build_prompt_for_all_sentences(sentences, samples):
    contexts = []

    for index, sample in enumerate(samples, start=1):
        contexts.append(
            f"CONTEXT {index}:\n{sample}"
        )

    context_text = "\n\n".join(contexts)

    sentence_text = []

    for index, sentence in enumerate(sentences, start=1):
        sentence_text.append(
            f"{index}. {sentence}"
        )

    sentences_text = "\n".join(sentence_text)

    return f"""
You are evaluating hallucinations in a generated passage.

For every generated sentence, compare it with all sampled
contexts.

A sentence is:

YES = supported by the sampled context
NO = not supported by the sampled context

Generated sentences:

{sentences_text}

Sampled contexts:

{context_text}

For EACH generated sentence, give exactly one score:

0 = supported
1 = not supported

Return ONLY a numbered list.

Example:

1. 0
2. 1
3. 0
4. 0

There must be exactly {len(sentences)} answers.
Do not provide explanations.
""".strip()


def parse_prompt_response(response, expected_count):
    import re

    matches = re.findall(
        r"\b([01])\b",
        response
    )

    if len(matches) == expected_count:
        return [float(value) for value in matches]

    return [0.5] * expected_count


def run_prompt_score(sentences, samples):
    if not sentences or not samples:
        return {
            "sentence_scores": [],
            "passage_score": 0.0
        }

    prompt = build_prompt_for_all_sentences(
        sentences,
        samples
    )

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0,
                max_tokens=max(100, len(sentences) * 5)
            )

            content = (
                response.choices[0].message.content
                or ""
            )

            scores = parse_prompt_response(
                content,
                len(sentences)
            )

            sentence_scores = [
                round(score, 4)
                for score in scores
            ]

            passage_score = (
                mean(sentence_scores)
                if sentence_scores
                else 0.0
            )

            return {
                "sentence_scores": sentence_scores,
                "passage_score": round(
                    passage_score,
                    4
                )
            }

        except Exception as error:

            error_text = str(error)

            if "429" not in error_text:
                raise

            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    "Groq rate limit reached after "
                    "multiple retries."
                ) from error

            wait_time = RETRY_WAIT_SECONDS * attempt

            print(
                f"Rate limit reached. "
                f"Waiting {wait_time} seconds..."
            )

            time.sleep(wait_time)

    raise RuntimeError(
        "Prompt scorer could not complete."
    )


# ============================================================
# VALIDATE SCORES
# ============================================================

def validate_scores(result, expected_sentence_count):

    sentence_scores = result.get(
        "sentence_scores",
        []
    )

    passage_score = result.get(
        "passage_score",
        0.0
    )

    if len(sentence_scores) != expected_sentence_count:
        raise ValueError(
            "Sentence score count mismatch: "
            f"expected {expected_sentence_count}, "
            f"got {len(sentence_scores)}"
        )

    for score in sentence_scores:

        if not 0.0 <= score <= 1.0:
            raise ValueError(
                f"Invalid sentence score: {score}"
            )

    if not 0.0 <= passage_score <= 1.0:
        raise ValueError(
            f"Invalid passage score: {passage_score}"
        )


# ============================================================
# MAIN BENCHMARK
# ============================================================

async def run_benchmark():

    prompts = load_test_prompts()

    print("=" * 60)
    print("SelfCheckGPT Hallucination Detection Benchmark")
    print("=" * 60)

    print(f"Prompts: {len(prompts)}")
    print(f"Samples per prompt: {NUM_SAMPLES}")
    print(f"Model: {MODEL_NAME}")

    print("=" * 60)

    generator = LLMGenerator(
        get_model_name()
    )

    evaluator = SelfCheckEvaluator()

    all_results = []

    for prompt_index, item in enumerate(
        prompts,
        start=1
    ):

        prompt_id = item.get(
            "id",
            prompt_index
        )

        prompt_text = item.get(
            "prompt",
            ""
        )

        topic = item.get(
            "topic",
            ""
        )

        expected_focus = item.get(
            "expected_focus",
            ""
        )

        print()
        print(
            f"[{prompt_index}/{len(prompts)}] "
            f"{topic}"
        )

        print("-" * 60)

        try:

            # ------------------------------------------------
            # GENERATE MAIN RESPONSE
            # ------------------------------------------------

            passage = await generator.generate_main_response(
                prompt_text
            )

            if not passage:
                raise RuntimeError(
                    "Empty generated passage."
                )

            # ------------------------------------------------
            # GENERATE SAMPLES
            # ------------------------------------------------

            samples = (
                await generator.generate_stochastic_samples(
                    prompt_text,
                    NUM_SAMPLES
                )
            )

            if not samples:
                raise RuntimeError(
                    "No sampled passages generated."
                )

            # ------------------------------------------------
            # PREPROCESS
            # ------------------------------------------------

            sentences = preprocess_response(
                passage
            )

            if not sentences:
                raise RuntimeError(
                    "No sentences found."
                )

            print(
                f"Generated sentences: "
                f"{len(sentences)}"
            )

            print(
                f"Generated samples: "
                f"{len(samples)}"
            )

            # ------------------------------------------------
            # UNIGRAM
            # ------------------------------------------------

            print(
                "Running Unigram scorer..."
            )

            unigram_result = evaluator.evaluate(
                sentences=sentences,
                samples=samples,
                method="unigram"
            )

            validate_scores(
                unigram_result,
                len(sentences)
            )

            print(
                "Unigram passage score: "
                f"{unigram_result['passage_score']:.4f}"
            )

            # ------------------------------------------------
            # PROMPT
            # ------------------------------------------------

            print(
                "Running Prompt scorer..."
            )

            prompt_result = run_prompt_score(
                sentences,
                samples
            )

            validate_scores(
                prompt_result,
                len(sentences)
            )

            print(
                "Prompt passage score: "
                f"{prompt_result['passage_score']:.4f}"
            )

            # ------------------------------------------------
            # NLI
            # ------------------------------------------------

            print(
                "Running NLI scorer..."
            )

            nli_result = evaluator.evaluate(
                sentences=sentences,
                samples=samples,
                method="nli"
            )

            validate_scores(
                nli_result,
                len(sentences)
            )

            print(
                "NLI passage score: "
                f"{nli_result['passage_score']:.4f}"
            )

            # ------------------------------------------------
            # SAVE RESULT
            # ------------------------------------------------

            result = {
                "id": prompt_id,
                "topic": topic,
                "prompt": prompt_text,
                "expected_focus": expected_focus,
                "generated_passage": passage,
                "generated_samples": samples,
                "num_sentences": len(sentences),
                "scores": {
                    "unigram": unigram_result,
                    "prompt": prompt_result,
                    "nli": nli_result
                }
            }

            all_results.append(result)

            print(
                "✓ Benchmark completed "
                "for this prompt."
            )

            # Save after every completed prompt
            save_results(
                prompts,
                all_results
            )

        except Exception as error:

            print(
                f"ERROR: {error}"
            )

            print(
                "Stopping benchmark."
            )

            break

    # --------------------------------------------------------
    # FINAL SAVE
    # --------------------------------------------------------

    save_results(
        prompts,
        all_results
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    print(
        f"Completed prompts: "
        f"{len(all_results)}/{len(prompts)}"
    )

    if all_results:

        unigram_scores = [
            item["scores"]["unigram"]["passage_score"]
            for item in all_results
        ]

        prompt_scores = [
            item["scores"]["prompt"]["passage_score"]
            for item in all_results
        ]

        nli_scores = [
            item["scores"]["nli"]["passage_score"]
            for item in all_results
        ]

        print(
            f"Average Unigram score: "
            f"{mean(unigram_scores):.4f}"
        )

        print(
            f"Average Prompt score: "
            f"{mean(prompt_scores):.4f}"
        )

        print(
            f"Average NLI score: "
            f"{mean(nli_scores):.4f}"
        )

    print()
    print(
        f"Results saved to: {RESULTS_FILE}"
    )

    print("=" * 60)


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(prompts, results):

    output = {
        "model": MODEL_NAME,
        "num_prompts": len(prompts),
        "num_samples": NUM_SAMPLES,
        "completed_prompts": len(results),
        "results": results
    }

    with open(
        RESULTS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    asyncio.run(
        run_benchmark()
    )