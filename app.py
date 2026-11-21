import hashlib
import io
import json

import openai
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from services.openai_service import (
    MODEL,
    extract_document_from_image,
    format_ocr_as_markdown,
)
from services.validation import (
    validate_document_content,
    validate_document_structure,
)

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
    for key in (
        "ocr_data",
        "ocr_markdown",
        "ocr_filename",
        "ocr_profile",
        "ocr_key_values",
        "ocr_tables",
        "active_request_hash",
    ):
        st.session_state.pop(key, None)

    st.rerun()


# =========================================
# Image uploader
# =========================================

with st.sidebar:
    st.header("Upload Document")

    st.header("Upload Document")

    document_profile = st.selectbox(
        "Document type", ["General document", "Receipt", "Invoice", "Form", "Letter"]
    )

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

    if st.session_state.get("active_request_hash") != request_hash:
        for key in (
            "ocr_data",
            "ocr_markdown",
            "ocr_filename",
            "ocr_profile",
            "ocr_key_values",
            "ocr_tables",
        ):
            st.session_state.pop(key, None)

        st.session_state["active_request_hash"] = request_hash

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
                st.session_state["ocr_key_values"] = document_data.get("key_values", [])
                st.session_state["ocr_tables"] = document_data.get("tables", [])

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

    key_values = st.session_state["ocr_key_values"]

    if key_values:
        key_value_df = pd.DataFrame(key_values)

        st.dataframe(key_value_df, use_container_width=True, hide_index=True)
    else:
        st.info("No key-value fields were extracted.")

    # =========================================
    # Display extracted tables
    # =========================================

    st.subheader("Extracted Tables")

    tables = st.session_state["ocr_tables"]

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
        st.json(st.session_state["ocr_data"])

    st.divider()

    # =================================
    # Download results
    # =================================

    markdown_col, json_col = st.columns(2)

    with markdown_col:
        st.download_button(
            label="Download Markdown",
            data=st.session_state["ocr_markdown"],
            file_name="ocr_result.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with json_col:
        json_result = json.dumps(
            st.session_state["ocr_data"], ensure_ascii=False, indent=2
        )

        st.download_button(
            label="Download JSON",
            data=json_result,
            file_name="ocr_result.json",
            mime="application/json",
            use_container_width=True,
        )

else:
    st.info("Upload a document image and click 'Extract Structured Text' to begin.")


# =========================================
# Footer
# =========================================

st.divider()

st.caption("Built with Python, Streamlit, and the OpenAI Responses API.")
