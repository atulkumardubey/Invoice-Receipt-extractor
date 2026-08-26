"""FastAPI backend exposing the invoice extraction pipeline over HTTP for the React UI."""
from dataclasses import dataclass

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend import extract_invoice

app = FastAPI(title="Invoice / Receipt Field Extractor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@dataclass
class _UploadedFileAdapter:
    """Mimics the Streamlit UploadedFile interface expected by backend.extract_invoice."""

    name: str
    _data: bytes

    def getvalue(self) -> bytes:
        return self._data


@app.post("/api/extract")
async def extract(file: UploadFile = File(...), provider: str = Form("nvidia")):
    data = await file.read()
    adapter = _UploadedFileAdapter(name=file.filename, _data=data)

    try:
        result = extract_invoice(adapter, provider=provider)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "filename": file.filename,
        "provider": provider,
        "raw_text": result.raw_text,
        "fields": result.fields.model_dump(mode="json") if result.fields else None,
        "raw_json": result.raw_json,
        "errors": result.errors,
    }
