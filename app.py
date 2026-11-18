import io

import openai
import streamlit as st
from PIL import Image

from services.openai_service import extract_text_from_image

# Page configuration
st.set_page_config(
    page_title="OpenAI OCR",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Application title
st.title("📄 OpenAI OCR")

st.markdown("Extract structured text from images using an OpenAI vision model.")

st.caption("Model: GPT-6 Luna")

st.markdown("---")

# Clear button
col1, col2 = st.columns([6, 1])

with col2:
    if st.button("Clear 🗑️"):
        st.session_state.pop("ocr_result", None)
        st.session_state.pop("ocr_filename", None)

        st.rerun()

# Image uploader
with st.sidebar:
    st.header("Upload Image")

    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Retrieve image bytes
        image_bytes = uploaded_file.getvalue()

        mime_type = uploaded_file.type

        # Display image preview
        try:
            image = Image.open(io.BytesIO(image_bytes))

            st.image(image, caption=uploaded_file.name, use_container_width=True)

        except (Image.UnidentifiedImageError, OSError):
            st.error("The uploaded file is not a valid image.")

            st.stop()

        # Extract text button
        if st.button("Extract Text 🔍", type="primary"):
            with st.spinner("Analyzing image with OpenAI..."):
                try:
                    # Call our OpenAI service
                    extracted_text = extract_text_from_image(
                        image_bytes=image_bytes, mime_type=mime_type
                    )

                    # Store result in session state
                    st.session_state["ocr_result"] = extracted_text

                    st.session_state["ocr_filename"] = uploaded_file.name

                    st.success("Text extraction completed!")

                # Error handling
                except ValueError as error:
                    st.error(str(error))

                except RuntimeError as error:
                    st.error(str(error))

                except openai.AuthenticationError:
                    st.error("Authentication failed. Check your OpenAI API key.")

                except openai.RateLimitError:
                    st.error(
                        "API rate or usage limit reached. "
                        "Check your OpenAI API usage and billing."
                    )

                except openai.APIConnectionError:
                    st.error(
                        "Could not connect to OpenAI. Check your internet connection."
                    )

                except openai.APIStatusError as error:
                    st.error(f"OpenAI API error: HTTP {error.status_code}")

                    st.caption(str(error.message))

                except openai.APIError as error:
                    st.error(f"OpenAI SDK error: {error}")


# Display extracted text
if "ocr_result" in st.session_state:
    st.subheader("Extracted Text")

    st.caption("Source image: " + st.session_state.get("ocr_filename", "Unknown"))

    st.markdown(st.session_state["ocr_result"])

    # Download extracted text
    st.download_button(
        label="Download OCR Result",
        data=st.session_state["ocr_result"],
        file_name="extracted_text.md",
        mime="text/markdown",
    )
else:
    st.info("Upload an image and click 'Extract Text' to see the results here.")


# Footer
st.markdown("---")

st.caption("Made with Python, Streamlit, and the OpenAI API.")
