"""Python backend: read file -> LLM extract -> self-verify -> Pydantic validate."""
import json
import os
import re
from dataclasses import dataclass, field
from io import BytesIO
from typing import Optional

import ollama
import pdfplumber
import pytesseract  # requires the Tesseract-OCR binary to be installed separately
from openai import OpenAI
from PIL import Image
from pydantic import ValidationError

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from models import InvoiceFields

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\atul.kumar.dubey\AppData\Local\Tesseract-OCR\tesseract.exe"

OLLAMA_MODEL = "llama3.1:8b"
NVIDIA_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

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
    try:
        return pytesseract.image_to_string(image).strip()
    except pytesseract.TesseractNotFoundError:
        raise ValueError(
            "Tesseract-OCR is not installed or not in PATH. "
            "Download the installer from https://github.com/UB-Mannheim/tesseract/wiki, "
            "install it, then restart the server. PDF files work without Tesseract."
        )


def _extract_json(raw: str) -> dict:
    """Pull a JSON object out of an LLM response, tolerating markdown fences."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"Model did not return JSON:\n{raw}")
    return json.loads(match.group(0))


def _chat_ollama(prompt: str) -> dict:
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            format="json",
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not reach Ollama model '{OLLAMA_MODEL}'. Is `ollama serve` running and "
            f"have you pulled the model (`ollama pull {OLLAMA_MODEL}`)?"
        ) from exc
    return _extract_json(response["message"]["content"])


def _chat_nvidia(prompt: str) -> dict:
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY not set. Add it to the .env file.")
    client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key)
    try:
        completion = client.chat.completions.create(
            model=NVIDIA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            top_p=0.95,
            max_tokens=1024,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    except Exception as exc:
        raise RuntimeError(f"NVIDIA API call failed: {exc}") from exc
    return _extract_json(completion.choices[0].message.content)


def _chat(prompt: str, provider: str = "nvidia") -> dict:
    if provider == "ollama":
        return _chat_ollama(prompt)
    return _chat_nvidia(prompt)


def llm_extract(text: str, provider: str = "nvidia") -> dict:
    prompt = EXTRACT_PROMPT.format(fields=", ".join(FIELDS), text=text)
    return _chat(prompt, provider)


def self_verify(raw_json: dict, text: str, provider: str = "nvidia") -> dict:
    prompt = VERIFY_PROMPT.format(
        extracted=json.dumps(raw_json), fields=", ".join(FIELDS), text=text
    )
    return _chat(prompt, provider)


def extract_invoice(uploaded_file, provider: str = "nvidia") -> ExtractionResult:
    text = read_file(uploaded_file)
    raw_json = llm_extract(text, provider)
    verified_json = self_verify(raw_json, text, provider)

    try:
        fields = InvoiceFields.model_validate(verified_json)
        errors = []
    except ValidationError as exc:
        fields = None
        errors = [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()]

    return ExtractionResult(raw_text=text, fields=fields, raw_json=verified_json, errors=errors)
