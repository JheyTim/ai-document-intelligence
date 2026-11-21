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

    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ").strip()


def format_ocr_as_markdown(document: dict) -> str:

    title = document.get("title") or "OCR Result"

    lines = [
        f"# {title}",
        "",
        f"**Document type:** {document.get('document_type', 'unknown')}",
        f"**Language:** {document.get('language', '')}",
        "",
        "## Full transcription",
        "",
    ]

    full_text = document.get("full_text", "").strip()

    if full_text:
        safe_text = full_text.replace("```", "'''\u200b")
        lines.extend(["```text", safe_text, "```", ""])
    else:
        lines.extend(["_No readable text detected._", ""])

    # Key-value fields
    key_values = document.get("key_values", [])

    if key_values:
        lines.extend(["## Key information", "", "| Field | Value |", "|---|---|"])

        for item in key_values:
            label = _markdown_cell(item.get("label", ""))
            value = _markdown_cell(item.get("value", ""))

            lines.append(f"| {label} | {value} |")

        lines.append("")

    # Tables
    tables = document.get("tables", [])

    if tables:
        lines.extend(["## Extracted tables", ""])

        for index, table in enumerate(tables, start=1):
            table_title = table.get("title") or f"Table {index}"

            lines.extend([f"### {table_title}", ""])

            headers = table.get("headers", [])
            rows = table.get("rows", [])

            column_count = max(len(headers), max((len(row) for row in rows), default=0))

            if column_count == 0:
                lines.extend(["_No table cells detected._", ""])
                continue

            # Add neutral headings if a row has more
            # columns than the supplied header list.
            normalized_headers = list(headers)

            while len(normalized_headers) < column_count:
                normalized_headers.append(f"Column {len(normalized_headers) + 1}")

            normalized_headers = normalized_headers[:column_count]

            lines.append(
                "| "
                + " | ".join(_markdown_cell(item) for item in normalized_headers)
                + " |"
            )

            lines.append("| " + " | ".join(["---"] * column_count) + " |")

            for row in rows:
                normalized_row = list(row)

                while len(normalized_row) < column_count:
                    normalized_row.append("")

                normalized_row = normalized_row[:column_count]

                lines.append(
                    "| "
                    + " | ".join(_markdown_cell(item) for item in normalized_row)
                    + " |"
                )

            lines.append("")

    # Uncertain text
    uncertain_items = document.get("uncertain_items", [])

    if uncertain_items:
        lines.extend(["## Uncertain or unreadable content", ""])

        for item in uncertain_items:
            lines.append(f"- {item}")

        lines.append("")

    return "\n".join(lines)
