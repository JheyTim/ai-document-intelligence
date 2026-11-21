# Llama OCR

A Streamlit app that extracts text and structured information from document
images using OpenAI's Responses API.

## Features

- Extracts a full transcription, key-value fields, tables, and uncertain text.
- Includes profiles for general documents, receipts, invoices, forms, and letters.
- Checks the returned data against a JSON schema and reports content warnings.
- Displays results as tables, formatted Markdown, and structured JSON.
- Downloads results as TXT, Markdown, or JSON files.

Supported image formats: PNG, JPG/JPEG, and WEBP.

## Requirements

- Python 3.11 or later
- An OpenAI API key
- An OpenAI model available to your account that supports image input and
  structured JSON output through the Responses API

## Setup

Create a virtual environment, activate it, and install the pinned dependencies.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS or Linux:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` in the project root (`Copy-Item .env.example .env`
in PowerShell, or `cp .env.example .env` on macOS/Linux), then set both values:

```dotenv
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=your_vision_model
```

The app loads `.env` from the project root and cannot start without
`OPENAI_MODEL`. Replace the placeholder with a model name available to your
OpenAI account. Uploaded images are sent to OpenAI for extraction. Keep your
API key private; `.env` is ignored by Git.

## Run

From the project root, start Streamlit:

```sh
streamlit run app.py
```

In the app, choose a document type, upload an image, and select **Extract
Structured Text**. Review the extracted fields, tables, Markdown, and JSON, then
use the download buttons to export the result. Selecting another image or
document type clears the previous extraction.

## Project layout

```text
app.py                      Streamlit interface and session state
services/openai_service.py  OpenAI request, schema, and Markdown formatting
services/prompts.py         Shared and document-specific OCR instructions
services/validation.py      Structure and content validation
exports/document_export.py  TXT, JSON, and filename export helpers
```
