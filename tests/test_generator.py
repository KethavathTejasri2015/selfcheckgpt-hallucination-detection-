import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from generator import LLMGenerator


def create_mock_response(content: str):
    response = MagicMock()
    response.choices[0].message.content = content
    return response


@pytest.mark.asyncio
async def test_generate_main_response():
    with patch("generator.get_api_key", return_value="test-api-key"):
        generator = LLMGenerator(
            model_name="openai/gpt-oss-20b"
        )

        generator.client.chat.completions.create = AsyncMock(
            return_value=create_mock_response(
                "This is a test response."
            )
        )

        result = await generator.generate_main_response(
            "What is Python?"
        )

        assert result == "This is a test response."

        generator.client.chat.completions.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_sample():
    with patch("generator.get_api_key", return_value="test-api-key"):
        generator = LLMGenerator(
            model_name="openai/gpt-oss-20b"
        )

        generator.client.chat.completions.create = AsyncMock(
            return_value=create_mock_response(
                "This is a sample response."
            )
        )

        result = await generator.generate_sample(
            "What is Python?"
        )

        assert result == "This is a sample response."


@pytest.mark.asyncio
async def test_generate_stochastic_samples():
    with patch("generator.get_api_key", return_value="test-api-key"):
        generator = LLMGenerator(
            model_name="openai/gpt-oss-20b"
        )

        generator.client.chat.completions.create = AsyncMock(
            side_effect=[
                create_mock_response("Sample 1"),
                create_mock_response("Sample 2"),
                create_mock_response("Sample 3"),
                create_mock_response("Sample 4"),
            ]
        )

        results = await generator.generate_stochastic_samples(
            "Test prompt",
            4,
        )

        assert len(results) == 4
        assert results == [
            "Sample 1",
            "Sample 2",
            "Sample 3",
            "Sample 4",
        ]


def test_invalid_sample_count():
    with patch("generator.get_api_key", return_value="test-api-key"):
        generator = LLMGenerator(
            model_name="openai/gpt-oss-20b"
        )

        with pytest.raises(ValueError):
            import asyncio

            asyncio.run(
                generator.generate_stochastic_samples(
                    "Test prompt",
                    3,
                )
            )