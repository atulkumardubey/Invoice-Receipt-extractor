"""Python backend: read file -> LLM extract -> self-verify -> Pydantic validate."""
import json
import re
from dataclasses import dataclass, field
from io import BytesIO
from typing import Optional

import ollama
import pdfplumber
import pytesseract  # requires the Tesseract-OCR binary to be installed separately
from PIL import Image
from pydantic import ValidationError

from models import InvoiceFields

MODEL_NAME = "llama3.1:8b"
FIELDS = list(InvoiceFields.model_fields.keys())

EXTRACT_PROMPT = """You are an expert at reading invoices and receipts.
Extract the following fields from the document text below and return ONLY a JSON object
with exactly these keys: {fields}.
Rules:
- invoice_date must be returned in ISO format (YYYY-MM-DD), no matter what format it appears in the document.
- subtotal, tax and total must be plain numbers (no currency symbols, no thousands separators).
- If a field is genuinely not present in the document, use null. Never guess a value.

Document text:
---
{text}
---
JSON:"""

VERIFY_PROMPT = """You extracted the following JSON from an invoice/receipt:
{extracted}

Re-check every field against the original document text below. Fix any value that does not
match the source, and set any field to null if it cannot be found in the text. Return ONLY the
corrected JSON object with exactly these keys: {fields}.

Document text:
---
{text}
---
Corrected JSON:"""


@dataclass
class ExtractionResult:
    raw_text: str
    fields: Optional[InvoiceFields]
    raw_json: dict
    errors: list = field(default_factory=list)


def read_file(uploaded_file) -> str:
    """Pull raw text out of an uploaded PDF/PNG/JPG file."""
    name = uploaded_file.name.lower()
    data = uploaded_file.getvalue()

    if name.endswith(".pdf"):
        text_parts = []
        with pdfplumber.open(BytesIO(data)) as pdf:
            for page in pdf.pages:
                text_parts.append(page.extract_text() or "")
        text = "\n".join(text_parts).strip()
        if text:
            return text
        raise ValueError(
            "No selectable text found in this PDF. Scanned PDFs aren't supported yet - "
            "please upload the page as a PNG/JPG instead."
        )

    image = Image.open(BytesIO(data))
    return pytesseract.image_to_string(image).strip()


def _extract_json(raw: str) -> dict:
    """Pull a JSON object out of an LLM response, tolerating markdown fences."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"Model did not return JSON:\n{raw}")
    return json.loads(match.group(0))


def _chat(prompt: str) -> dict:
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            format="json",
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not reach Ollama model '{MODEL_NAME}'. Is `ollama serve` running and "
            f"have you pulled the model (`ollama pull {MODEL_NAME}`)?"
        ) from exc
    return _extract_json(response["message"]["content"])


def llm_extract(text: str) -> dict:
    prompt = EXTRACT_PROMPT.format(fields=", ".join(FIELDS), text=text)
    return _chat(prompt)


def self_verify(raw_json: dict, text: str) -> dict:
    prompt = VERIFY_PROMPT.format(
        extracted=json.dumps(raw_json), fields=", ".join(FIELDS), text=text
    )
    return _chat(prompt)


def extract_invoice(uploaded_file) -> ExtractionResult:
    text = read_file(uploaded_file)
    raw_json = llm_extract(text)
    verified_json = self_verify(raw_json, text)

    try:
        fields = InvoiceFields.model_validate(verified_json)
        errors = []
    except ValidationError as exc:
        fields = None
        errors = [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()]

    return ExtractionResult(raw_text=text, fields=fields, raw_json=verified_json, errors=errors)
