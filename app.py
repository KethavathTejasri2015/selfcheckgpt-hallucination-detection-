
import asyncio
import time
import html

import streamlit as st

from config import get_model_name
from generator import LLMGenerator
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

generator = LLMGenerator(
    model_name=get_model_name()
)

evaluator = SelfCheckEvaluator()


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

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.1
    )


with col2:

    samples = st.slider(
        "Number of Samples",
        min_value=4,
        max_value=20,
        value=5,
        step=1
    )


with col3:

    method = st.selectbox(
        "Evaluation Method",
        [
            "NLI",
            "Prompt-based",
            "N-gram"
        ]
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
# Highlighting function
# ---------------------------------------------------------

def display_highlighted_response(sentences, scores):

    highlighted_sentences = []

    for sentence, score in zip(sentences, scores):

        safe_sentence = html.escape(sentence)

        if score > 0.66:
            css_class = "high"

        elif score >= 0.33:
            css_class = "medium"

        else:
            css_class = "low"

        highlighted_sentences.append(
            f'<span class="sentence {css_class}" '
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
        f"Temperature: {temperature:.1f}  |  "
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

        st.success(
            "🟢 Low likelihood of hallucination"
        )

    elif passage_score <= 0.66:

        st.warning(
            "🟡 Moderate likelihood of hallucination"
        )

    else:

        st.error(
            "🔴 High likelihood of hallucination"
        )

    # -----------------------------------------------------
    # Highlighted response
    # -----------------------------------------------------

    st.subheader("Generated Response")

    display_highlighted_response(
        sentences,
        sentence_scores
    )

    # -----------------------------------------------------
    # Score table
    # -----------------------------------------------------

    st.subheader("Sentence Scores")

    for index, (sentence, score) in enumerate(
        zip(sentences, sentence_scores),
        start=1
    ):

        if score > 0.66:
            label = "🔴 High"

        elif score >= 0.33:
            label = "🟡 Moderate"

        else:
            label = "🟢 Low"

        st.write(
            f"**Sentence {index} — {label} — "
            f"Score: {score:.2f}**"
        )

        st.write(sentence)

    # -----------------------------------------------------
    # Raw response
    # -----------------------------------------------------

    with st.expander(
        "View plain generated response"
    ):

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

            st.markdown(
                f"**Sample {i}**"
            )

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

                execution_time = (
                    time.time() - start_time
                )

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

                scoring_method = method_map[
                    method
                ]

                # ---------------------------------------------
                # Generate response
                # ---------------------------------------------

                with st.spinner(
                    "Generating response and stochastic samples..."
                ):

                    result = asyncio.run(
                        generator.generate_all(
                            prompt,
                            samples
                        )
                    )

                response = result["response"]

                sentences = result["sentences"]

                stochastic_samples = result["samples"]

                # ---------------------------------------------
                # Evaluation
                # ---------------------------------------------

                with st.spinner(
                    f"Calculating {method} hallucination scores..."
                ):

                    if scoring_method == "prompt":

                        st.warning(
                            "Prompt-based scoring requires an "
                            "evaluator callback from the backend. "
                            "Please use NLI or N-gram unless the "
                            "prompt evaluator is configured."
                        )

                        st.stop()

                    evaluation = evaluator.evaluate(
                        sentences=sentences,
                        samples=stochastic_samples,
                        method=scoring_method
                    )

                sentence_scores = (
                    evaluation["sentence_scores"]
                )

                passage_score = (
                    evaluation["passage_score"]
                )

                execution_time = (
                    time.time() - start_time
                )

                display_results(
                    response=response,
                    sentences=sentences,
                    stochastic_samples=stochastic_samples,
                    sentence_scores=sentence_scores,
                    passage_score=passage_score,
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

            with st.expander(
                "Technical error details"
            ):

                st.exception(e)
