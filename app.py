import hashlib
import io

import openai
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from exports.document_export import (
    document_to_json,
    document_to_txt,
    make_export_filename,
)
from services.openai_service import (
    MODEL,
    extract_document_from_image,
    format_ocr_as_markdown,
)
from services.prompts import DOCUMENT_PROFILES
from services.validation import (
    validate_document_content,
    validate_document_structure,
)

OCR_RESULT_KEYS = (
    "ocr_data",
    "ocr_markdown",
    "ocr_filename",
    "ocr_profile",
)
ACTIVE_REQUEST_HASH_KEY = "active_request_hash"


def clear_ocr_results(*, include_request_hash: bool = False) -> None:
    keys = OCR_RESULT_KEYS
    if include_request_hash:
        keys += (ACTIVE_REQUEST_HASH_KEY,)

    for key in keys:
        st.session_state.pop(key, None)


# =========================================
# Page configuration
# =========================================

st.set_page_config(page_title="OpenAI OCR", page_icon="📄", layout="wide")


# =========================================
# Application heading
# =========================================

st.title("📄 AI Document Intelligence")
st.write("Extract document text, key information, and tables using OpenAI.")

st.caption(f"Vision model: {MODEL}")

st.divider()


# =========================================
# Clear previous results
# =========================================

if st.button("Clear results 🗑️"):
    clear_ocr_results(include_request_hash=True)
    st.rerun()


# =========================================
# Image uploader
# =========================================

with st.sidebar:
    st.header("Upload Document")

    document_profile = st.selectbox("Document type", options=list(DOCUMENT_PROFILES))

    uploaded_file = st.file_uploader(
        "Choose an image", type=["png", "jpg", "jpeg", "webp"]
    )


if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()

    # Identify the actual image format
    try:
        preview = Image.open(io.BytesIO(image_bytes))

        image_format = preview.format

        preview.verify()

        format_to_mime = {
            "JPEG": "image/jpeg",
            "PNG": "image/png",
            "WEBP": "image/webp",
        }

        mime_type = format_to_mime.get(image_format)

        if mime_type is None:
            st.error("Unsupported image format.")
            st.stop()

        # Reopen after verify() for image display
        preview = Image.open(io.BytesIO(image_bytes))

    except (UnidentifiedImageError, OSError):
        st.error("The uploaded file is not a valid image.")
        st.stop()

    # Clear stale results if another file is selected
    request_hash = hashlib.sha256(
        image_bytes + b"\0" + document_profile.encode("utf-8")
    ).hexdigest()

    if st.session_state.get(ACTIVE_REQUEST_HASH_KEY) != request_hash:
        clear_ocr_results()
        st.session_state[ACTIVE_REQUEST_HASH_KEY] = request_hash

    # Display image preview
    st.sidebar.subheader("Image Preview")

    st.sidebar.image(preview, caption=uploaded_file.name, use_container_width=True)

    st.sidebar.caption(f"File size: {len(image_bytes):,} bytes")

    # =====================================
    # Run structured OCR
    # =====================================

    if st.sidebar.button("Extract Structured Text 🔍", type="primary"):
        with st.spinner("Analyzing document with OpenAI..."):
            try:
                # Get structured JSON data
                document_data = extract_document_from_image(
                    image_bytes=image_bytes,
                    mime_type=mime_type,
                    document_profile=document_profile,
                )

                # Check JSON structure against our schema
                validation_errors = validate_document_structure(document_data)

                if validation_errors:
                    error_details = "\n".join(validation_errors)

                    raise RuntimeError(
                        "The extracted document failed validation:\n" + error_details
                    )

                # Check additional content rules
                validation_warnings = validate_document_content(document_data)

                if validation_warnings:
                    st.warning(
                        "The Document was extracted, but some content checks raised warnings."
                    )

                    for warning in validation_warnings:
                        st.write(f"- {warning}")
                else:
                    st.success("Document structure and content checks passed.")

                # Convert the structured data to Markdown
                markdown_result = format_ocr_as_markdown(document_data)

                # Store results
                st.session_state["ocr_data"] = document_data
                st.session_state["ocr_markdown"] = markdown_result
                st.session_state["ocr_filename"] = uploaded_file.name
                st.session_state["ocr_profile"] = document_profile

                st.success("Document extraction completed!")

            except ValueError as error:
                st.error(str(error))

            except RuntimeError as error:
                st.error(str(error))

            except openai.AuthenticationError:
                st.error("Authentication failed. Check your OpenAI API key.")

            except openai.RateLimitError:
                st.error(
                    "API rate or usage limit reached. Check your API usage and billing."
                )

            except openai.APIConnectionError:
                st.error("Could not connect to OpenAI. Check your internet connection.")

            except openai.APIStatusError as error:
                st.error(f"OpenAI API error: HTTP {error.status_code}")

                st.caption(str(error.message))

            except openai.APIError as error:
                st.error(f"OpenAI SDK error: {error}")


if "ocr_data" in st.session_state:
    # =========================================
    # Display extracted key-value fields
    # =========================================
    st.subheader("Extracted Key-Value Fields")

    document_data = st.session_state["ocr_data"]
    
    key_values = document_data.get("key_values", [])

    if key_values:
        key_value_df = pd.DataFrame(key_values)

        st.dataframe(key_value_df, use_container_width=True, hide_index=True)
    else:
        st.info("No key-value fields were extracted.")

    # =========================================
    # Display extracted tables
    # =========================================

    st.subheader("Extracted Tables")

    tables = document_data.get("tables", [])

    if tables:
        for table_index, table in enumerate(tables, start=1):
            table_title = table.get("title") or f"Table {table_index}"

            st.markdown(f"### {table_title}")

            headers = table.get("headers", [])
            rows = table.get("rows", [])

            if headers:
                table_df = pd.DataFrame(rows, columns=headers)

                st.dataframe(table_df, use_container_width=True, hide_index=True)

            else:
                st.warning(f"{table_title} has no extracted headers.")
    else:
        st.info("No tables were detected.")

    st.divider()

    # =========================================
    # Display OCR results
    # =========================================
    st.subheader("OCR Results")

    st.caption(f"Source document: {st.session_state['ocr_filename']}")
    st.caption(f"Extraction profile: {st.session_state.get('ocr_profile', 'Unknown')}")

    markdown_tab, json_tab = st.tabs(["Formatted Markdown", "Structured JSON"])

    # Markdown display
    with markdown_tab:
        st.markdown(st.session_state["ocr_markdown"])

    # JSON display
    with json_tab:
        st.json(document_data)

    st.divider()

    # =================================
    # Download results
    # =================================

    st.subheader("Download Extracted Document")

    txt_content = document_to_txt(document_data)
    markdown_content = st.session_state["ocr_markdown"]
    json_content = document_to_json(document_data)

    txt_filename = make_export_filename(document_data, "txt")
    markdown_filename = make_export_filename(document_data, "md")
    json_filename = make_export_filename(document_data, "json")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.download_button(
            label="Download TXT",
            data=txt_content,
            file_name=txt_filename,
            mime="text/plain",
            key="download_txt",
        )

    with col2:
        st.download_button(
            label="Download Markdown",
            data=markdown_content,
            file_name=markdown_filename,
            mime="text/markdown",
            key="download_markdown",
        )
    with col3:
        st.download_button(
            label="Download JSON",
            data=json_content,
            file_name=json_filename,
            mime="application/json",
            key="download_json",
        )


else:
    st.info("Upload a document image and click 'Extract Structured Text' to begin.")


# =========================================
# Footer
# =========================================

st.divider()

st.caption("Built with Python, Streamlit, and the OpenAI Responses API.")
