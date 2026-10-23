import asyncio
import html
import time

import streamlit as st

from config import get_model_name
from eval_ngram import tokenize
from integration import SelfCheckPipeline
from scorer import SelfCheckEvaluator


st.set_page_config(
    page_title="SelfCheckGPT",
    page_icon="🔍",
    layout="wide"
)


# ---------------------------------------------------------
# Page title
# ---------------------------------------------------------

st.title("🔍 SelfCheckGPT – Hallucination Detector")

st.write(
    "Detect potentially unsupported statements in generated responses."
)


# ---------------------------------------------------------
# Backend components
# ---------------------------------------------------------
# Only the evaluator is cached. It holds the heavy NLI model, which
# must not be reloaded on every Streamlit rerun.
#
# The pipeline (async Groq client + semaphore) is created per click,
# because each asyncio.run() call uses a fresh event loop and async
# clients must not be reused across loops.

@st.cache_resource(show_spinner=False)
def get_evaluator() -> SelfCheckEvaluator:
    return SelfCheckEvaluator()


def build_pipeline() -> SelfCheckPipeline:
    pipeline = SelfCheckPipeline(
        model_name=get_model_name()
    )
    pipeline.evaluator = get_evaluator()
    return pipeline


# ---------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------

st.markdown(
    """
    <style>

    .highlight-box {
        padding: 20px;
        border-radius: 10px;
        line-height: 2.2;
        font-size: 17px;
        background-color: #111111;
        border: 1px solid #333333;
    }

    .sentence {
        padding: 4px 7px;
        border-radius: 5px;
        margin: 2px;
    }

    .low {
        background-color: #1b5e20;
        color: white;
    }

    .medium {
        background-color: #8a6d1d;
        color: white;
    }

    .high {
        background-color: #8b1e1e;
        color: white;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# Input
# ---------------------------------------------------------

st.divider()

st.subheader("Input")

prompt = st.text_area(
    "Enter your prompt:",
    placeholder="Example: Explain how the human heart works.",
    height=120
)


col1, col2, col3 = st.columns(3)


with col1:

    samples = st.slider(
        "Number of Samples",
        min_value=4,
        max_value=20,
        value=5,
        step=1,
        help="Stochastic samples are generated at temperature 1.0."
    )


with col2:

    method = st.selectbox(
        "Evaluation Method",
        [
            "NLI",
            "Prompt-based",
            "N-gram"
        ]
    )


with col3:

    threshold = st.slider(
        "Safe Answer Threshold",
        min_value=0.2,
        max_value=0.9,
        value=0.66,
        step=0.05,
        help="Sentences scoring above this are removed from the Safe Answer."
    )


# ---------------------------------------------------------
# Demo mode
# ---------------------------------------------------------

st.divider()

demo_mode = st.checkbox(
    "🎬 Demo Mode",
    help="Use precomputed results without calling the Groq API."
)


# ---------------------------------------------------------
# Demo data
# ---------------------------------------------------------

DEMO_RESPONSE = (
    "The human heart is a muscular organ that pumps blood "
    "throughout the body. It has four chambers that help "
    "maintain continuous blood circulation. The heart also "
    "receives oxygen directly from the lungs through a special "
    "blood vessel."
)

DEMO_SENTENCES = [
    "The human heart is a muscular organ that pumps blood throughout the body.",
    "It has four chambers that help maintain continuous blood circulation.",
    "The heart also receives oxygen directly from the lungs through a special blood vessel."
]

DEMO_SAMPLES = [
    "The heart is a muscular organ responsible for pumping blood around the body.",
    "The human heart contains four chambers and circulates blood continuously.",
    "Blood is pumped through the body by the muscular heart.",
    "The heart has four chambers that support blood circulation.",
    "The heart pumps blood to different parts of the body."
]

DEMO_SCORES = [
    0.12,
    0.38,
    0.82
]


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def risk_class(score: float) -> str:
    if score > 0.66:
        return "high"
    if score >= 0.33:
        return "medium"
    return "low"


def risk_label(score: float) -> str:
    if score > 0.66:
        return "🔴 High"
    if score >= 0.33:
        return "🟡 Moderate"
    return "🟢 Low"


# ---------------------------------------------------------
# Highlighting function
# ---------------------------------------------------------

def display_highlighted_response(sentences, scores):

    highlighted_sentences = []

    for sentence, score in zip(sentences, scores):

        safe_sentence = html.escape(sentence)

        highlighted_sentences.append(
            f'<span class="sentence {risk_class(score)}" '
            f'title="Hallucination score: {score:.2f}">'
            f'{safe_sentence}'
            f'</span>'
        )

    highlighted_text = " ".join(
        highlighted_sentences
    )

    st.markdown(
        f"""
        <div class="highlight-box">
            {highlighted_text}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption(
        "🟢 Low: < 0.33   |   "
        "🟡 Moderate: 0.33–0.66   |   "
        "🔴 High: > 0.66"
    )


# ---------------------------------------------------------
# Safe Answer (mitigation step, not part of the original paper)
# ---------------------------------------------------------

def display_safe_answer(sentences, scores, limit):

    kept = [
        s for s, sc in zip(sentences, scores) if sc <= limit
    ]

    removed = [
        (s, sc) for s, sc in zip(sentences, scores) if sc > limit
    ]

    st.subheader("Safe Answer")

    st.caption(
        f"Sentences scoring above {limit:.2f} are removed. "
        f"Kept {len(kept)} of {len(sentences)}."
    )

    if kept:
        st.success(" ".join(kept))
    else:
        st.warning(
            "Every sentence was flagged. "
            "No safe answer can be produced at this threshold."
        )

    if removed:
        with st.expander(f"View {len(removed)} removed sentence(s)"):
            for sentence, score in removed:
                st.write(f"🔴 **{score:.2f}** — {sentence}")


# ---------------------------------------------------------
# Evidence view (explains WHY a sentence was flagged)
# ---------------------------------------------------------

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "on",
    "to", "and", "or", "it", "its", "that", "this", "for", "with",
    "as", "by", "at", "from", "be", "has", "have", "had", "also"
}


def content_words(text):
    return {t for t in tokenize(text) if t not in STOPWORDS}


def support_ratio(sentence, sample):
    words = content_words(sentence)
    if not words:
        return 0.0
    return len(words & content_words(sample)) / len(words)


def display_evidence(sentences, scores, stochastic_samples, limit):

    flagged = [
        (s, sc) for s, sc in zip(sentences, scores) if sc > limit
    ]

    st.subheader("Evidence")

    if not flagged:
        st.success("No sentence was flagged, so there is nothing to explain.")
        return

    st.caption(
        "A sample 'supports' a sentence if it shares at least 50% of the "
        "sentence's key words (a simple word-overlap check)."
    )

    for sentence, score in flagged:

        ratios = [
            support_ratio(sentence, sample)
            for sample in stochastic_samples
        ]

        supporting = sum(1 for r in ratios if r >= 0.5)
        best_index = max(range(len(ratios)), key=lambda i: ratios[i])

        st.error(f"🔴 {sentence}  (score {score:.2f})")

        st.write(
            f"**{supporting} of {len(stochastic_samples)}** samples "
            f"support this sentence."
        )

        st.progress(supporting / len(stochastic_samples))

        with st.expander("Closest sample"):
            st.caption(
                f"Sample {best_index + 1} — "
                f"{ratios[best_index]:.0%} word overlap"
            )
            st.write(stochastic_samples[best_index])


# ---------------------------------------------------------
# Result display function
# ---------------------------------------------------------

def display_results(
    response,
    sentences,
    stochastic_samples,
    sentence_scores,
    passage_score,
    method_name,
    execution_time
):

    st.success("Analysis completed successfully!")

    st.subheader("Result")

    st.info(
        f"Method: {method_name}  |  "
        f"Samples: {len(stochastic_samples)}"
    )

    # -----------------------------------------------------
    # Overall score
    # -----------------------------------------------------

    st.subheader("Overall Hallucination Score")

    st.metric(
        "Passage Score",
        f"{passage_score:.2f}"
    )

    if passage_score < 0.33:
        st.success("🟢 Low likelihood of hallucination")

    elif passage_score <= 0.66:
        st.warning("🟡 Moderate likelihood of hallucination")

    else:
        st.error("🔴 High likelihood of hallucination")

    # -----------------------------------------------------
    # Highlighted response
    # -----------------------------------------------------

    st.subheader("Generated Response")

    display_highlighted_response(
        sentences,
        sentence_scores
    )

    # -----------------------------------------------------
    # Safe answer
    # -----------------------------------------------------

    display_safe_answer(
        sentences,
        sentence_scores,
        threshold
    )

    # -----------------------------------------------------
    # Evidence
    # -----------------------------------------------------

    display_evidence(
        sentences,
        sentence_scores,
        stochastic_samples,
        threshold
    )

    # -----------------------------------------------------
    # Score table
    # -----------------------------------------------------

    st.subheader("Sentence Scores")

    for index, (sentence, score) in enumerate(
        zip(sentences, sentence_scores),
        start=1
    ):

        st.write(
            f"**Sentence {index} — {risk_label(score)} — "
            f"Score: {score:.2f}**"
        )

        st.write(sentence)

    # -----------------------------------------------------
    # Raw response
    # -----------------------------------------------------

    with st.expander("View plain generated response"):
        st.write(response)

    # -----------------------------------------------------
    # Stochastic samples
    # -----------------------------------------------------

    with st.expander(
        f"View {len(stochastic_samples)} stochastic samples"
    ):

        for i, sample in enumerate(
            stochastic_samples,
            start=1
        ):

            st.markdown(f"**Sample {i}**")

            st.write(sample)

            if i < len(stochastic_samples):
                st.divider()

    # -----------------------------------------------------
    # Execution time
    # -----------------------------------------------------

    st.caption(
        f"Execution time: {execution_time:.2f} seconds"
    )


# ---------------------------------------------------------
# Main button
# ---------------------------------------------------------

if st.button(
    "🔎 Check Response",
    type="primary"
):

    if not prompt.strip():

        st.warning(
            "Please enter a prompt first."
        )

    else:

        start_time = time.time()

        try:

            # =================================================
            # DEMO MODE
            # =================================================

            if demo_mode:

                with st.spinner(
                    "Loading precomputed demo result..."
                ):

                    time.sleep(0.5)

                    response = DEMO_RESPONSE
                    sentences = DEMO_SENTENCES
                    stochastic_samples = DEMO_SAMPLES
                    sentence_scores = DEMO_SCORES

                    passage_score = round(
                        sum(sentence_scores)
                        / len(sentence_scores),
                        4
                    )

                execution_time = time.time() - start_time

                display_results(
                    response=response,
                    sentences=sentences,
                    stochastic_samples=stochastic_samples,
                    sentence_scores=sentence_scores,
                    passage_score=passage_score,
                    method_name=f"{method} (Demo)",
                    execution_time=execution_time
                )

            # =================================================
            # LIVE MODE
            # =================================================

            else:

                method_map = {
                    "NLI": "nli",
                    "Prompt-based": "prompt",
                    "N-gram": "unigram"
                }

                scoring_method = method_map[method]

                pipeline = build_pipeline()

                with st.spinner(
                    f"Generating response and scoring with {method}..."
                ):

                    result = asyncio.run(
                        pipeline.generate_and_score(
                            prompt=prompt,
                            num_samples=samples,
                            method=scoring_method
                        )
                    )

                execution_time = time.time() - start_time

                display_results(
                    response=result["response"],
                    sentences=result["sentences"],
                    stochastic_samples=result["samples"],
                    sentence_scores=result["sentence_scores"],
                    passage_score=result["passage_score"],
                    method_name=method,
                    execution_time=execution_time
                )

        except Exception as e:

            st.error(
                "An error occurred while generating "
                "or evaluating the response."
            )

            st.error(
                "Please check your API connection and "
                "backend configuration."
            )

            with st.expander("Technical error details"):
                st.exception(e)