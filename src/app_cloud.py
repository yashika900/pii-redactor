"""
Streamlit Community Cloud entry point.

Identical interface to app.py, but uses PIIRedactorCloud (llama-cpp-python)
instead of PIIRedactor (Ollama), since Streamlit Community Cloud doesn't
support system-level installs like Ollama.

For local, fully-offline usage with Ollama, see app.py instead.
"""

import streamlit as st

from inference.llamacpp_redactor import PIIRedactorCloud

st.set_page_config(page_title="Local PII Redactor", page_icon="🔒", layout="wide")


@st.cache_resource
def get_redactor() -> PIIRedactorCloud:
    with st.spinner("Loading model (first run only, ~1-2 minutes)..."):
        return PIIRedactorCloud()


def main() -> None:
    st.title("🔒 Local PII Redactor")
    st.caption(
        "A fine-tuned 3B model + regex safety net, running fully on-device "
        "inference (no data sent to any third-party LLM API). Hosted here "
        "for demo convenience — see the "
        "[GitHub repo](https://github.com/yashika900/pii-redactor) for "
        "fully offline local setup instructions."
    )

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
            with st.spinner("Processing..."):
                result = redactor.redact(input_text)

            st.text_area("Result:", value=result.final_output, height=300)
            st.success(f"Processed in {result.elapsed_seconds:.2f} seconds")

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
        "Served via llama-cpp-python"
    )


if __name__ == "__main__":
    main()
