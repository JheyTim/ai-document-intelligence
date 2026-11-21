import json

from jsonschema import Draft202012Validator

from .openai_service import OCR_SCHEMA


# Validate the schema itseft
def validate_ocr_schema():
    Draft202012Validator.check_schema(OCR_SCHEMA)

    return True


# Validate extracted JSON structure
def validate_document_structure(document_data: dict) -> list[str]:
    errors = []

    validator = Draft202012Validator(OCR_SCHEMA)

    for error in validator.iter_errors(document_data):
        path = ".".join(str(part) for part in error.absolute_path)

        location = path if path else "root"

        errors.append(f"{location}: {error.message}")

    return errors


# Validate business rules
def validate_document_content(document_data: dict) -> list[str]:
    warnings = []

    if not document_data.get("title"):
        warnings.append("No document title was extracted.")

    if not document_data.get("full_text"):
        warnings.append("No readable full transcription was returned.")

    for index, item in enumerate(document_data.get("key_values", [])):
        if not item.get("label", "").strip():
            warnings.append(f"Key-value item {index + 1} has an empty label.")

    return warnings


# Validate JSON text
def validate_json_text(json_text: str) -> tuple[dict | None, list[str]]:

    try:
        document_data = json.loads(json_text)

    except json.JSONDecodeError as error:
        return None, [f"Invalid JSON: {error}"]

    if not isinstance(document_data, dict):
        return None, ["The JSON root must be an object."]

    errors = validate_document_structure(document_data)

    if errors:
        return None, errors

    return document_data, []


def validate_table_consistency(document_data: dict) -> list[str]:
    """
    Check whether every table row has the same number
    of cells as the table has headers.
    """
    errors = []

    for table_index, table in enumerate(document_data.get("tables", [])):
        headers = table.get("headers", [])
        rows = table.get("rows", [])

        expected_cells = len(headers)

        for row_index, row in enumerate(rows):
            actual_cells = len(row)

            if actual_cells != expected_cells:
                errors.append(
                    f"Table {table_index + 1}, "
                    f"row {row_index + 1}: expected "
                    f"{expected_cells} cells, but found "
                    f"{actual_cells}."
                )

    return errors
