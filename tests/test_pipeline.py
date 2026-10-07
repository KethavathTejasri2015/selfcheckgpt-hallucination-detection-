import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------
# Mock NLI module
# ---------------------------------------------------------
# This prevents the tests from requiring the heavy
# transformers package or downloading an NLI model.
mock_nli = MagicMock()
mock_nli.NLIScorer = MagicMock()
sys.modules["eval_nli"] = mock_nli


from preprocessor import preprocess_response
from scorer import SelfCheckEvaluator
from integration import SelfCheckPipeline


# =========================================================
# PREPROCESSOR TESTS
# =========================================================

def test_preprocess_normal_sentences():
    text = "Python is easy to learn. It is widely used in AI."

    result = preprocess_response(text)

    assert result == [
        "Python is easy to learn.",
        "It is widely used in AI.",
    ]


def test_preprocess_empty_text():
    result = preprocess_response("")

    assert result == []


def test_preprocess_whitespace():
    result = preprocess_response("   ")

    assert result == []


def test_preprocess_invalid_input():
    with pytest.raises(TypeError):
        preprocess_response(123)


# =========================================================
# SCORER TESTS
# =========================================================

def test_empty_inputs():
    evaluator = SelfCheckEvaluator()

    result = evaluator.evaluate(
        sentences=[],
        samples=[],
        method="unigram",
    )

    assert result["sentence_scores"] == []
    assert result["passage_score"] == 0.0


def test_whitespace_sentences():
    evaluator = SelfCheckEvaluator()

    result = evaluator.evaluate(
        sentences=["   ", "   "],
        samples=["This is a sample."],
        method="unigram",
    )

    assert result["sentence_scores"] == []
    assert result["passage_score"] == 0.0


def test_unigram_scores_are_in_range():
    evaluator = SelfCheckEvaluator()

    result = evaluator.evaluate(
        sentences=[
            "Python is a programming language."
        ],
        samples=[
            "Python is a programming language.",
            "Python is commonly used for programming.",
        ],
        method="unigram",
    )

    for score in result["sentence_scores"]:
        assert 0.0 <= score <= 1.0

    assert 0.0 <= result["passage_score"] <= 1.0


def test_passage_score_is_average():
    evaluator = SelfCheckEvaluator()

    import scorer

    original_function = scorer.score_unigram

    scorer.score_unigram = MagicMock(
        return_value=[0.2, 0.6, 0.8]
    )

    try:
        result = evaluator.evaluate(
            sentences=["A.", "B.", "C."],
            samples=["Sample."],
            method="unigram",
        )

        assert result["sentence_scores"] == [
            0.2,
            0.6,
            0.8,
        ]

        assert result["passage_score"] == 0.5333

    finally:
        scorer.score_unigram = original_function


def test_prompt_requires_evaluator():
    evaluator = SelfCheckEvaluator()

    with pytest.raises(ValueError):
        evaluator.evaluate(
            sentences=["Python is a language."],
            samples=[
                "Python is used for programming."
            ],
            method="prompt",
        )


def test_prompt_scorer_with_mock():
    evaluator = SelfCheckEvaluator()

    def mock_evaluator(prompt):
        return "Yes"

    result = evaluator.evaluate(
        sentences=["Python is a language."],
        samples=[
            "Python is a programming language."
        ],
        method="prompt",
        evaluator_fn=mock_evaluator,
    )

    for score in result["sentence_scores"]:
        assert 0.0 <= score <= 1.0

    assert 0.0 <= result["passage_score"] <= 1.0


def test_nli_scorer_with_mock():
    evaluator = SelfCheckEvaluator()

    mock_scorer = MagicMock()
    mock_scorer.score_sentence.return_value = 0.7

    evaluator._nli_scorer = mock_scorer

    result = evaluator.evaluate(
        sentences=[
            "Python is a programming language."
        ],
        samples=[
            "Python is commonly used for programming."
        ],
        method="nli",
    )

    assert result["sentence_scores"] == [0.7]
    assert result["passage_score"] == 0.7

    mock_scorer.score_sentence.assert_called_once()


def test_unknown_method():
    evaluator = SelfCheckEvaluator()

    with pytest.raises(ValueError):
        evaluator.evaluate(
            sentences=["Python is a language."],
            samples=["Python is useful."],
            method="invalid",
        )


# =========================================================
# INTEGRATION PIPELINE TESTS
# =========================================================

@pytest.mark.asyncio
async def test_pipeline_integration():

    # Prevent the test from requiring a real GROQ API key.
    with patch(
        "integration.get_api_key",
        return_value="test-api-key",
    ), patch(
        "generator.get_api_key",
        return_value="test-api-key",
    ):

        pipeline = SelfCheckPipeline(
            model_name="openai/gpt-oss-20b"
        )

        # generate_all() is an async function,
        # so it must be mocked with AsyncMock.
        pipeline.generator.generate_all = AsyncMock(
            return_value={
                "response": "Python is a programming language.",
                "sentences": [
                    "Python is a programming language."
                ],
                "samples": [
                    "Python is widely used for programming.",
                    "Python is a popular programming language.",
                ],
            }
        )

        # Mock scoring so no real model/API call happens.
        pipeline.evaluator.evaluate = MagicMock(
            return_value={
                "sentence_scores": [0.25],
                "passage_score": 0.25,
            }
        )

        result = await pipeline.generate_and_score(
            prompt="What is Python?",
            num_samples=4,
            method="unigram",
        )

        assert result["response"] != ""

        assert len(result["sentences"]) > 0

        assert len(result["samples"]) > 0

        assert len(result["sentence_scores"]) > 0

        assert 0.0 <= result["passage_score"] <= 1.0

        pipeline.generator.generate_all.assert_awaited_once()


@pytest.mark.asyncio
async def test_pipeline_invalid_method():

    with patch(
        "integration.get_api_key",
        return_value="test-api-key",
    ), patch(
        "generator.get_api_key",
        return_value="test-api-key",
    ):

        pipeline = SelfCheckPipeline(
            model_name="openai/gpt-oss-20b"
        )

        with pytest.raises(ValueError):
            await pipeline.generate_and_score(
                prompt="What is Python?",
                num_samples=4,
                method="invalid",
            )