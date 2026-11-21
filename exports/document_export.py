import json
import re


def document_to_txt(document_data: dict) -> str:
    lines = []

    title = document_data.get("title") or "Untitled Document"
    document_type = document_data.get("document_type", "")

    lines.append(title.upper())
    lines.append(f"Document Type: {document_type}")
    lines.append("")

    # Full transcription
    lines.append("FULL TRANSCRIPTION")
    lines.append("=" * 40)
    lines.append(document_data.get("full_text", ""))
    lines.append("")

    # Key-value fields
    lines.append("KEY-VALUE FIELDS")
    lines.append("=" * 40)

    for item in document_data.get("key_values", []):
        label = item.get("label", "")
        value = item.get("value", "")
        lines.append(f"{label}: {value}")

    lines.append("")

    # Tables
    lines.append("EXTRACTED TABLES")
    lines.append("=" * 40)

    for table_index, table in enumerate(document_data.get("tables", []), start=1):
        table_title = table.get("title") or f"Table {table_index}"

        lines.append(f"\nTABLE: {table_title}")

        headers = table.get("headers", [])
        rows = table.get("rows", [])

        if headers:
            lines.append(" | ".join(str(h) for h in headers))

        for row in rows:
            lines.append(" | ".join(str(cell) for cell in row))

    # Uncertain items
    lines.append("")
    lines.append("UNCERTAIN ITEMS")
    lines.append("=" * 40)

    uncertain_items = document_data.get("uncertain_items", [])

    if uncertain_items:
        for item in uncertain_items:
            lines.append(f"- {item}")
    else:
        lines.append("None recorded.")

    return "\n".join(lines)


def document_to_json(document_data: dict) -> str:
    return json.dumps(document_data, ensure_ascii=False, indent=2, allow_nan=False)


def make_export_filename(document_data: dict, extension: str) -> str:
    title = document_data.get("title") or "extracted_document"

    # Replace characters that are unsuitable for filenames.
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(title))

    stem = re.sub(r"\s+", "_", stem)
    stem = stem.strip(" ._")

    # Keep filenames reasonably short.
    stem = stem[:80] or "extracted_document"

    # Allow only alphanumeric extension characters.
    clean_extension = re.sub(r"[^a-zA-Z0-9]", "", extension.lstrip("."))

    if not clean_extension:
        raise ValueError("A valid file extension is required.")

    return f"{stem}.{clean_extension}"
