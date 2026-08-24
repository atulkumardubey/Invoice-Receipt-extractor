from datetime import date
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel


class InvoiceFields(BaseModel):
    invoice_no: str
    invoice_date: date  # any format in -> ISO out
    vendor: str
    subtotal: Decimal
    tax: Optional[Decimal] = None
    total: Decimal
    payment_terms: Optional[str] = None
