"""Integration tests for the FastAPI /api/extract endpoint using TestClient."""
import io
import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from backend import ExtractionResult
from models import InvoiceFields
from server import app

client = TestClient(app)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

SAMPLE_FIELDS = InvoiceFields(
    invoice_no="INV-001",
    invoice_date="2026-08-24",
    vendor="ACME Corp",
    subtotal="1000.00",
    tax="180.00",
    total="1180.00",
    payment_terms="Net 30",
)

SAMPLE_RESULT = ExtractionResult(
    raw_text="Invoice text from document",
    fields=SAMPLE_FIELDS,
    raw_json={"invoice_no": "INV-001"},
    errors=[],
)

FAILED_RESULT = ExtractionResult(
    raw_text="Invoice text",
    fields=None,
    raw_json={"invoice_no": "INV-001", "total": "bad"},
    errors=["total: value is not a valid decimal"],
)


def fake_pdf() -> tuple[str, io.BytesIO, str]:
    return ("invoice.pdf", io.BytesIO(b"%PDF fake"), "application/pdf")


def fake_image() -> tuple[str, io.BytesIO, str]:
    return ("receipt.png", io.BytesIO(b"PNG_BYTES"), "image/png")


# ---------------------------------------------------------------------------
# Successful extraction
# ---------------------------------------------------------------------------

class TestExtractEndpointSuccess:
    def test_pdf_nvidia_returns_200(self):
        with patch("server.extract_invoice", return_value=SAMPLE_RESULT):
            response = client.post(
                "/api/extract",
                data={"provider": "nvidia"},
                files={"file": fake_pdf()},
            )
        assert response.status_code == 200

    def test_response_contains_expected_keys(self):
        with patch("server.extract_invoice", return_value=SAMPLE_RESULT):
            response = client.post(
                "/api/extract",
                data={"provider": "nvidia"},
                files={"file": fake_pdf()},
            )
        body = response.json()
        assert set(body.keys()) == {"filename", "provider", "raw_text", "fields", "raw_json", "errors"}

    def test_filename_returned(self):
        with patch("server.extract_invoice", return_value=SAMPLE_RESULT):
            response = client.post(
                "/api/extract",
                data={"provider": "nvidia"},
                files={"file": fake_pdf()},
            )
        assert response.json()["filename"] == "invoice.pdf"

    def test_provider_echoed_back(self):
        with patch("server.extract_invoice", return_value=SAMPLE_RESULT):
            response = client.post(
                "/api/extract",
                data={"provider": "ollama"},
                files={"file": fake_pdf()},
            )
        assert response.json()["provider"] == "ollama"

    def test_fields_returned_when_extraction_succeeds(self):
        with patch("server.extract_invoice", return_value=SAMPLE_RESULT):
            response = client.post(
                "/api/extract",
                data={"provider": "nvidia"},
                files={"file": fake_pdf()},
            )
        fields = response.json()["fields"]
        assert fields["invoice_no"] == "INV-001"
        assert fields["vendor"] == "ACME Corp"

    def test_nvidia_is_default_provider(self):
        captured = {}

        def capture(adapter, provider="nvidia"):
            captured["provider"] = provider
            return SAMPLE_RESULT

        with patch("server.extract_invoice", side_effect=capture):
            client.post("/api/extract", files={"file": fake_pdf()})
        assert captured["provider"] == "nvidia"

    def test_image_upload_works(self):
        with patch("server.extract_invoice", return_value=SAMPLE_RESULT):
            response = client.post(
                "/api/extract",
                data={"provider": "nvidia"},
                files={"file": fake_image()},
            )
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Validation failure (Pydantic rejects LLM output)
# ---------------------------------------------------------------------------

class TestExtractEndpointValidationFailure:
    def test_fields_is_null_when_pydantic_fails(self):
        with patch("server.extract_invoice", return_value=FAILED_RESULT):
            response = client.post(
                "/api/extract",
                data={"provider": "nvidia"},
                files={"file": fake_pdf()},
            )
        assert response.status_code == 200
        body = response.json()
        assert body["fields"] is None
        assert len(body["errors"]) > 0


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

class TestExtractEndpointErrors:
    def test_missing_file_returns_422(self):
        response = client.post("/api/extract", data={"provider": "nvidia"})
        assert response.status_code == 422

    def test_backend_exception_returns_400(self):
        with patch("server.extract_invoice", side_effect=ValueError("No text in PDF")):
            response = client.post(
                "/api/extract",
                data={"provider": "nvidia"},
                files={"file": fake_pdf()},
            )
        assert response.status_code == 400
        assert "No text in PDF" in response.json()["detail"]

    def test_nvidia_api_error_returns_400(self):
        with patch("server.extract_invoice", side_effect=RuntimeError("NVIDIA API call failed")):
            response = client.post(
                "/api/extract",
                data={"provider": "nvidia"},
                files={"file": fake_pdf()},
            )
        assert response.status_code == 400

    def test_tesseract_missing_returns_400(self):
        with patch("server.extract_invoice", side_effect=ValueError("Tesseract-OCR is not installed")):
            response = client.post(
                "/api/extract",
                data={"provider": "nvidia"},
                files={"file": fake_image()},
            )
        assert response.status_code == 400
        assert "Tesseract" in response.json()["detail"]
