"""Unit tests for the backend pipeline: file reading, LLM calls, and extraction flow."""
import json
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

import backend
from backend import (
    _extract_json,
    extract_invoice,
    read_file,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SAMPLE_JSON = {
    "invoice_no": "INV-001",
    "invoice_date": "2026-08-24",
    "vendor": "ACME Corp",
    "subtotal": "1000.00",
    "tax": "180.00",
    "total": "1180.00",
    "payment_terms": "Net 30",
}
SAMPLE_JSON_STR = json.dumps(SAMPLE_JSON)


@dataclass
class FakeFile:
    name: str
    _data: bytes

    def getvalue(self) -> bytes:
        return self._data


def make_pdf_file(name="invoice.pdf") -> FakeFile:
    return FakeFile(name=name, _data=b"%PDF-1.4 fake content")


def make_image_file(name="invoice.png") -> FakeFile:
    return FakeFile(name=name, _data=b"PNG_BYTES")


def mock_pdfplumber(text: str):
    """Return a context-manager mock for pdfplumber.open that yields one page with `text`."""
    mock_page = MagicMock()
    mock_page.extract_text.return_value = text
    mock_pdf = MagicMock()
    mock_pdf.__enter__ = lambda s: s
    mock_pdf.__exit__ = MagicMock(return_value=False)
    mock_pdf.pages = [mock_page]
    return mock_pdf


# ---------------------------------------------------------------------------
# _extract_json
# ---------------------------------------------------------------------------

class TestExtractJson:
    def test_plain_json_object(self):
        result = _extract_json(SAMPLE_JSON_STR)
        assert result["invoice_no"] == "INV-001"
        assert result["vendor"] == "ACME Corp"

    def test_json_inside_markdown_fence(self):
        raw = f"```json\n{SAMPLE_JSON_STR}\n```"
        result = _extract_json(raw)
        assert result["total"] == "1180.00"

    def test_json_with_leading_text(self):
        raw = f"Here is the extracted data:\n{SAMPLE_JSON_STR}"
        result = _extract_json(raw)
        assert result["invoice_date"] == "2026-08-24"

    def test_no_json_raises_value_error(self):
        with pytest.raises(ValueError, match="Model did not return JSON"):
            _extract_json("I cannot find any invoice fields here.")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            _extract_json("")


# ---------------------------------------------------------------------------
# read_file — PDF
# ---------------------------------------------------------------------------

class TestReadFilePdf:
    def test_pdf_with_text_returns_text(self):
        with patch("backend.pdfplumber.open", return_value=mock_pdfplumber("Invoice text")):
            result = read_file(make_pdf_file())
        assert result == "Invoice text"

    def test_pdf_strips_whitespace(self):
        with patch("backend.pdfplumber.open", return_value=mock_pdfplumber("  text  ")):
            result = read_file(make_pdf_file())
        assert result == "text"

    def test_pdf_multiple_pages_joined(self):
        page1 = MagicMock(); page1.extract_text.return_value = "Page 1"
        page2 = MagicMock(); page2.extract_text.return_value = "Page 2"
        mock_pdf = MagicMock()
        mock_pdf.__enter__ = lambda s: s
        mock_pdf.__exit__ = MagicMock(return_value=False)
        mock_pdf.pages = [page1, page2]
        with patch("backend.pdfplumber.open", return_value=mock_pdf):
            result = read_file(make_pdf_file())
        assert "Page 1" in result and "Page 2" in result

    def test_pdf_without_text_raises(self):
        with patch("backend.pdfplumber.open", return_value=mock_pdfplumber("")):
            with pytest.raises(ValueError, match="No selectable text"):
                read_file(make_pdf_file())

    def test_pdf_none_page_text_treated_as_empty(self):
        mock_page = MagicMock(); mock_page.extract_text.return_value = None
        mock_pdf = MagicMock()
        mock_pdf.__enter__ = lambda s: s
        mock_pdf.__exit__ = MagicMock(return_value=False)
        mock_pdf.pages = [mock_page]
        with patch("backend.pdfplumber.open", return_value=mock_pdf):
            with pytest.raises(ValueError, match="No selectable text"):
                read_file(make_pdf_file())


# ---------------------------------------------------------------------------
# read_file — images
# ---------------------------------------------------------------------------

class TestReadFileImage:
    def test_png_returns_ocr_text(self):
        with patch("backend.Image.open", return_value=MagicMock()), \
             patch("backend.pytesseract.image_to_string", return_value="  OCR text  "):
            result = read_file(make_image_file("invoice.png"))
        assert result == "OCR text"

    def test_jpg_returns_ocr_text(self):
        with patch("backend.Image.open", return_value=MagicMock()), \
             patch("backend.pytesseract.image_to_string", return_value="Receipt text"):
            result = read_file(make_image_file("receipt.jpg"))
        assert result == "Receipt text"

    def test_jpeg_extension_works(self):
        with patch("backend.Image.open", return_value=MagicMock()), \
             patch("backend.pytesseract.image_to_string", return_value="data"):
            result = read_file(make_image_file("scan.jpeg"))
        assert result == "data"

    def test_tesseract_not_found_raises_friendly_error(self):
        import pytesseract as pt
        with patch("backend.Image.open", return_value=MagicMock()), \
             patch("backend.pytesseract.image_to_string", side_effect=pt.TesseractNotFoundError()):
            with pytest.raises(ValueError, match="Tesseract-OCR is not installed"):
                read_file(make_image_file())


# ---------------------------------------------------------------------------
# _chat_nvidia
# ---------------------------------------------------------------------------

class TestChatNvidia:
    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="NVIDIA_API_KEY not set"):
            backend._chat_nvidia("any prompt")

    def test_successful_call_returns_dict(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
        mock_completion = MagicMock()
        mock_completion.choices[0].message.content = SAMPLE_JSON_STR
        with patch("backend.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = mock_completion
            result = backend._chat_nvidia("extract invoice")
        assert result["invoice_no"] == "INV-001"

    def test_api_exception_raises_runtime_error(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
        with patch("backend.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.side_effect = Exception("timeout")
            with pytest.raises(RuntimeError, match="NVIDIA API call failed"):
                backend._chat_nvidia("extract invoice")

    def test_thinking_disabled_in_request(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
        mock_completion = MagicMock()
        mock_completion.choices[0].message.content = SAMPLE_JSON_STR
        with patch("backend.OpenAI") as mock_openai:
            mock_client = mock_openai.return_value
            mock_client.chat.completions.create.return_value = mock_completion
            backend._chat_nvidia("extract invoice")
            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


# ---------------------------------------------------------------------------
# _chat_ollama
# ---------------------------------------------------------------------------

class TestChatOllama:
    def test_successful_call_returns_dict(self):
        with patch("backend.ollama.chat") as mock_chat:
            mock_chat.return_value = {"message": {"content": SAMPLE_JSON_STR}}
            result = backend._chat_ollama("extract invoice")
        assert result["vendor"] == "ACME Corp"

    def test_connection_error_raises_runtime_error(self):
        with patch("backend.ollama.chat", side_effect=Exception("Connection refused")):
            with pytest.raises(RuntimeError, match="Could not reach Ollama"):
                backend._chat_ollama("extract invoice")

    def test_uses_json_format(self):
        with patch("backend.ollama.chat") as mock_chat:
            mock_chat.return_value = {"message": {"content": SAMPLE_JSON_STR}}
            backend._chat_ollama("extract invoice")
            call_kwargs = mock_chat.call_args.kwargs
        assert call_kwargs.get("format") == "json"


# ---------------------------------------------------------------------------
# extract_invoice — full pipeline
# ---------------------------------------------------------------------------

class TestExtractInvoice:
    def test_happy_path_nvidia(self):
        with patch("backend.read_file", return_value="raw invoice text"), \
             patch("backend._chat", return_value=SAMPLE_JSON):
            result = extract_invoice(make_pdf_file(), provider="nvidia")
        assert result.fields is not None
        assert result.fields.invoice_no == "INV-001"
        assert result.fields.vendor == "ACME Corp"
        assert result.errors == []
        assert result.raw_text == "raw invoice text"

    def test_happy_path_ollama(self):
        with patch("backend.read_file", return_value="raw invoice text"), \
             patch("backend._chat", return_value=SAMPLE_JSON):
            result = extract_invoice(make_pdf_file(), provider="ollama")
        assert result.fields is not None
        assert result.errors == []

    def test_null_optional_fields_accepted(self):
        json_with_nulls = {**SAMPLE_JSON, "tax": None, "payment_terms": None}
        with patch("backend.read_file", return_value="text"), \
             patch("backend._chat", return_value=json_with_nulls):
            result = extract_invoice(make_pdf_file())
        assert result.fields is not None
        assert result.fields.tax is None
        assert result.fields.payment_terms is None

    def test_pydantic_failure_returns_errors_not_raise(self):
        bad_json = {**SAMPLE_JSON, "invoice_date": "not-a-date", "total": "bad"}
        with patch("backend.read_file", return_value="text"), \
             patch("backend._chat", return_value=bad_json):
            result = extract_invoice(make_pdf_file())
        assert result.fields is None
        assert len(result.errors) > 0

    def test_self_verify_called_after_extract(self):
        with patch("backend.read_file", return_value="text"), \
             patch("backend._chat", return_value=SAMPLE_JSON) as mock_chat:
            extract_invoice(make_pdf_file(), provider="nvidia")
        # _chat must be called twice: once for extract, once for verify
        assert mock_chat.call_count == 2

    def test_provider_passed_to_chat(self):
        with patch("backend.read_file", return_value="text"), \
             patch("backend._chat", return_value=SAMPLE_JSON) as mock_chat:
            extract_invoice(make_pdf_file(), provider="ollama")
        for call in mock_chat.call_args_list:
            assert call.args[1] == "ollama" or call.kwargs.get("provider") == "ollama"

    def test_raw_json_preserved_in_result(self):
        with patch("backend.read_file", return_value="text"), \
             patch("backend._chat", return_value=SAMPLE_JSON):
            result = extract_invoice(make_pdf_file())
        assert result.raw_json == SAMPLE_JSON
