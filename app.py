import streamlit as st
import time

st.set_page_config(
    page_title="SelfCheckGPT",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 SelfCheckGPT – Hallucination Detector")
st.write("Detect potentially unsupported statements in generated responses.")

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
        ["NLI", "Prompt-based", "N-gram"]
    )

st.divider()

if st.button("🔎 Check Response", type="primary"):

    if not prompt.strip():
        st.warning("Please enter a prompt first.")

    else:
        start_time = time.time()

        with st.spinner("Analyzing response..."):
            time.sleep(1)

        st.success("Analysis completed!")

        st.subheader("Result")

        st.info(
            f"Method: {method}  |  "
            f"Temperature: {temperature}  |  "
            f"Samples: {samples}"
        )

        st.subheader("Generated Response")

        response = (
            "The human heart is a muscular organ that pumps blood "
            "throughout the body. It has four chambers that help "
            "maintain blood circulation. The heart was discovered "
            "in the twentieth century by a famous scientist."
        )

        st.write(response)

        execution_time = time.time() - start_time

        st.caption(f"Execution time: {execution_time:.2f} seconds")