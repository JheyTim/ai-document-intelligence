import base64
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from .prompts import build_ocr_instructions

# =========================================
# Configuration
# =========================================

BASE_DIR = Path(__file__).resolve().parents[1]

load_dotenv(BASE_DIR / ".env")

MODEL = os.getenv("OPENAI_MODEL")

if not MODEL:
    raise RuntimeError("OPENAI_MODEL is not configured in .env")

ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}


# =========================================
# Structured OCR JSON Schema
# =========================================

OCR_SCHEMA = {
    "type": "object",
    "properties": {
        "document_type": {
            "type": "string",
            "enum": [
                "receipt",
                "invoice",
                "form",
                "letter",
                "report",
                "table",
                "other",
                "unknown",
            ],
        },
        "title": {"type": "string"},
        "language": {"type": "string"},
        "full_text": {"type": "string"},
        "key_values": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["label", "value"],
                "additionalProperties": False,
            },
        },
        "tables": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "headers": {"type": "array", "items": {"type": "string"}},
                    "rows": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "string"}},
                    },
                },
                "required": ["title", "headers", "rows"],
                "additionalProperties": False,
            },
        },
        "uncertain_items": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "document_type",
        "title",
        "language",
        "full_text",
        "key_values",
        "tables",
        "uncertain_items",
    ],
    "additionalProperties": False,
}


# =========================================
# Extract structured data from an image
# =========================================


def extract_document_from_image(
    image_bytes: bytes, mime_type: str, document_profile: str = "General document"
) -> dict:

    if mime_type not in ALLOWED_MIME_TYPES:
        raise ValueError(f"Unsupported image type: {mime_type}")

    if not image_bytes:
        raise ValueError("The uploaded image is empty.")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY was not found. Check your .env file.")

    # Encode image bytes as Base64
    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    # Build the data URL
    image_data_url = f"data:{mime_type};base64,{base64_image}"

    # Initialize the client
    client = OpenAI()

    # Request structured output
    response = client.responses.create(
        model=MODEL,
        instructions=build_ocr_instructions(document_profile),
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Transcribe and extract the information from this document."
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": image_data_url,
                        "detail": "original",
                    },
                ],
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "document_ocr",
                "strict": True,
                "schema": OCR_SCHEMA,
            }
        },
    )

    # Check that the response completed
    if response.status != "completed":
        raise RuntimeError(
            f"OCR response did not complete. "
            f"Status: {response.status}. "
            f"Details: {response.incomplete_details}"
        )

    # Detect a model refusal
    for output_item in response.output:
        if output_item.type == "message":
            for content_item in output_item.content:
                if content_item.type == "refusal":
                    raise RuntimeError(
                        "The model refused the request: " + content_item.refusal
                    )

    # Read the JSON response
    output = response.output_text

    if not output:
        raise RuntimeError("The model returned an empty response.")

    try:
        document_data = json.loads(output)
    except json.JSONDecodeError as error:
        raise RuntimeError("The model response could not be parsed as JSON.") from error

    return document_data


# =========================================
# Convert structured OCR data to Markdown
# =========================================


def _markdown_cell(value: str) -> str:
    """Prepare a value for a Markdown table cell."""

    return (
        str(value)
        .replace("|", r"\|")
        .replace("\r", " ")
        .replace("\r\n", "\n")
        .replace("\n", "<br>")
    )


def format_ocr_as_markdown(document_data: dict) -> str:
    lines = []

    title = document_data.get("title") or "Untitled Document"

    lines.append(f"# {title}")
    lines.append("")

    lines.append(f"**Document Type:** {document_data.get('document_type', '')}")
    lines.append("")

    # Full transcription
    lines.append("## Full Transcription")
    lines.append("")
    lines.append(document_data.get("full_text", ""))
    lines.append("")

    # Key-value fields
    lines.append("## Key-Value Fields")
    lines.append("")
    lines.append("| Label | Value |")
    lines.append("|---|---|")

    for item in document_data.get("key_values", []):
        label = _markdown_cell(item.get("label", ""))
        value = _markdown_cell(item.get("value", ""))

        lines.append(f"| {label} | {value} |")

    lines.append("")

    # Tables
    for table_index, table in enumerate(document_data.get("tables", []), start=1):
        table_title = table.get("title") or f"Table {table_index}"

        headers = table.get("headers", [])
        rows = table.get("rows", [])

        lines.append(f"## {table_title}")
        lines.append("")

        width = max([len(headers)] + [len(row) for row in rows])

        if width == 0:
            lines.append("_No table cells were extracted._")
            lines.append("")
            continue

        display_headers = list(headers)

        while len(display_headers) < width:
            display_headers.append(f"Unlabeled column {len(display_headers) + 1}")

        header_text = [_markdown_cell(header) for header in display_headers]

        lines.append("| " + " | ".join(header_text) + " |")
        lines.append("| " + " | ".join(["---"] * width) + " |")

        for row in rows:
            cells = list(row)

            while len(cells) < width:
                cells.append("")

            cell_text = [_markdown_cell(cell) for cell in cells]

            lines.append("| " + " | ".join(cell_text) + " |")

        lines.append("")

    # Uncertain items
    lines.append("## Uncertain Items")
    lines.append("")

    uncertain_items = document_data.get("uncertain_items", [])

    if uncertain_items:
        for item in uncertain_items:
            lines.append(f"- {item}")
    else:
        lines.append("None recorded.")

    return "\n".join(lines)
