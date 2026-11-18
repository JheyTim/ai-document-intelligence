import os
import base64
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# -----------------------------------------
# 1. Configuration
# -----------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

load_dotenv(BASE_DIR / ".env")

MODEL = "gpt-6-luna"

ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
}


# -----------------------------------------
# 2. OCR prompt
# -----------------------------------------

OCR_PROMPT = """
You are an OCR assistant.

Analyze the provided document image.

Your instructions:

1. Extract all readable text.
2. Preserve the original wording.
3. Preserve the logical reading order.
4. Use Markdown formatting where appropriate.
5. Preserve tables as Markdown tables when their structure
   is clear.
6. Do not summarize the document.
7. Do not invent missing or unreadable information.
8. Mark unclear text as [Unclear].

Return the extracted document content.
"""


# -----------------------------------------
# 3. Extract text from an image
# -----------------------------------------


def extract_text_from_image(image_bytes: bytes, mime_type: str) -> str:

    # Validate the image format
    if mime_type not in ALLOWED_MIME_TYPES:
        raise ValueError(f"Unsupported image type: {mime_type}")

    # Retrieve the API key
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY was not found. Check your .env file.")

    # Convert image bytes to Base64
    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    # Build the image data URL
    image_data_url = f"data:{mime_type};base64,{base64_image}"

    # Initialize the OpenAI client
    client = OpenAI(api_key=api_key)

    # Send the image to OpenAI
    response = client.responses.create(
        model=MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": OCR_PROMPT},
                    {"type": "input_image", "image_url": image_data_url},
                ],
            }
        ],
    )

    # Retrieve the generated text
    extracted_text = response.output_text

    if not extracted_text:
        raise RuntimeError("The model returned an empty response.")

    return extracted_text
