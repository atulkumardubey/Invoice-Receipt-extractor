"""Tests for InvoiceFields Pydantic model — type coercion, validation, and edge cases."""
from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from models import InvoiceFields

VALID = {
    "invoice_no": "INV-2026-001",
    "invoice_date": "2026-08-24",
    "vendor": "ACME Corp",
    "subtotal": "15000.00",
    "tax": "2700.00",
    "total": "17700.00",
    "payment_terms": "Net 30",
}


class TestInvoiceFieldsValid:
    def test_all_fields_accepted(self):
        f = InvoiceFields(**VALID)
        assert f.invoice_no == "INV-2026-001"
        assert f.invoice_date == date(2026, 8, 24)
        assert f.vendor == "ACME Corp"
        assert f.subtotal == Decimal("15000.00")
        assert f.tax == Decimal("2700.00")
        assert f.total == Decimal("17700.00")
        assert f.payment_terms == "Net 30"

    def test_optional_tax_is_none(self):
        f = InvoiceFields(**{**VALID, "tax": None})
        assert f.tax is None

    def test_optional_payment_terms_is_none(self):
        f = InvoiceFields(**{**VALID, "payment_terms": None})
        assert f.payment_terms is None

    def test_both_optional_fields_absent(self):
        data = {k: v for k, v in VALID.items() if k not in ("tax", "payment_terms")}
        f = InvoiceFields(**data)
        assert f.tax is None
        assert f.payment_terms is None

    def test_date_object_input_accepted(self):
        f = InvoiceFields(**{**VALID, "invoice_date": date(2026, 8, 24)})
        assert f.invoice_date == date(2026, 8, 24)

    def test_integer_amounts_accepted(self):
        f = InvoiceFields(**{**VALID, "subtotal": 15000, "tax": 2700, "total": 17700})
        assert f.subtotal == Decimal("15000")
        assert f.total == Decimal("17700")

    def test_zero_tax_accepted(self):
        f = InvoiceFields(**{**VALID, "tax": "0.00"})
        assert f.tax == Decimal("0.00")


class TestInvoiceFieldsInvalid:
    def test_bad_date_string_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            InvoiceFields(**{**VALID, "invoice_date": "24-Aug-2026"})
        assert "invoice_date" in str(exc_info.value)

    def test_non_numeric_total_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            InvoiceFields(**{**VALID, "total": "abc"})
        assert "total" in str(exc_info.value)

    def test_missing_invoice_no_raises(self):
        data = {k: v for k, v in VALID.items() if k != "invoice_no"}
        with pytest.raises(ValidationError) as exc_info:
            InvoiceFields(**data)
        assert "invoice_no" in str(exc_info.value)

    def test_missing_vendor_raises(self):
        data = {k: v for k, v in VALID.items() if k != "vendor"}
        with pytest.raises(ValidationError):
            InvoiceFields(**data)

    def test_missing_total_raises(self):
        data = {k: v for k, v in VALID.items() if k != "total"}
        with pytest.raises(ValidationError):
            InvoiceFields(**data)

    def test_currency_symbol_in_amount_raises(self):
        with pytest.raises(ValidationError):
            InvoiceFields(**{**VALID, "subtotal": "$15,000.00"})
