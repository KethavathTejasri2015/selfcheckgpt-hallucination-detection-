import math
import re
from collections import Counter


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def build_unigram_model(samples, alpha=1e-6):
    counts = Counter()

    for sample in samples:
        counts.update(tokenize(sample))

    total_count = sum(counts.values())
    vocabulary_size = len(counts)

    return counts, total_count, vocabulary_size, alpha


def token_probability(
    token,
    counts,
    total_count,
    vocabulary_size,
    alpha=1e-6
):
    return (
        counts[token] + alpha
    ) / (
        total_count + alpha * (vocabulary_size + 1)
    )


def score_sentence(sentence, samples):
    counts, total_count, vocabulary_size, alpha = (
        build_unigram_model(samples)
    )

    tokens = tokenize(sentence)

    if not tokens:
        return 0.0

    nll_scores = []

    for token in tokens:
        probability = token_probability(
            token,
            counts,
            total_count,
            vocabulary_size,
            alpha
        )

        nll_scores.append(-math.log(probability))

    return max(nll_scores)


def score_sentences(sentences, samples):
    return [
        score_sentence(sentence, samples)
        for sentence in sentences
    ]


if __name__ == "__main__":
    samples = [
        "The cat is sitting on the mat.",
        "The cat is sitting on the floor.",
        "The cat likes the mat."
    ]

    sentences = [
        "The cat is sitting on the mat.",
        "The elephant is flying to Mars."
    ]

    scores = score_sentences(sentences, samples)

    print(scores)