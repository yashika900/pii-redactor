"""
Streamlit web interface for the Local PII Redactor.

Paste text in, get redacted text out — all inference happens locally via
Ollama, with no data sent to any external cloud API.
"""

import streamlit as st

from inference.redactor import PIIRedactor

st.set_page_config(page_title="Local PII Redactor", page_icon="🔒", layout="wide")


@st.cache_resource
def get_redactor() -> PIIRedactor:
    return PIIRedactor(model_name="pii-redactor")


def main() -> None:
    st.title("🔒 Local PII Redactor")
    st.caption("Runs 100% offline on your device — no data sent to any cloud API")

    redactor = get_redactor()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input")
        input_text = st.text_area(
            "Paste text to redact:", height=300, placeholder="Paste your text here..."
        )
        redact_button = st.button("Redact PII", type="primary")

    with col2:
        st.subheader("Redacted Output")
        if redact_button and input_text.strip():
            with st.spinner("Processing locally..."):
                result = redactor.redact(input_text)

            st.text_area("Result:", value=result.final_output, height=300)
            st.success(
                f"Processed in {result.elapsed_seconds:.2f} seconds — fully offline"
            )

            if result.safety_net_triggered:
                st.info(
                    "Safety-net layer caught additional PII the model missed: "
                    f"{result.regex_safety_net_catches}"
                )
        elif redact_button:
            st.warning("Please paste some text first.")

    st.divider()
    st.caption(
        "Model: Fine-tuned Llama 3.2 3B (LoRA) | Quantized GGUF (Q4_K_M) | "
        "Served via Ollama"
    )


if __name__ == "__main__":
    main()
