import { useRef, useState } from "react";

const API_URL = "http://localhost:8000";

function formatSize(bytes) {
  return `${(bytes / 1024).toFixed(1)} KB`;
}

export default function App() {
  const [file, setFile] = useState(null);
  const [provider, setProvider] = useState("nvidia");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  function handleFileChange(event) {
    const picked = event.target.files?.[0] ?? null;
    setFile(picked);
    setResult(null);
    setError(null);
  }

  async function handleExtract() {
    if (!file) return;
    setLoading(true);
    setResult(null);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("provider", provider);
      const response = await fetch(`${API_URL}/api/extract`, {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Extraction failed.");
      }
      setResult(data);
    } catch (err) {
      setError(err.message || "Could not reach the extraction API.");
    } finally {
      setLoading(false);
    }
  }

  function handleDownload() {
    if (!result?.fields) return;
    const blob = new Blob([JSON.stringify(result.fields, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    const baseName = result.filename.replace(/\.[^/.]+$/, "");
    link.href = url;
    link.download = `${baseName}.json`;
    link.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="page">
      <h1>🧾 Invoice / Receipt Field Extractor</h1>
      <p className="caption">
        Upload in the browser, one prompt extracts the fields, Pydantic
        enforces every format, and the result renders on screen.
      </p>

      <hr />

      <div className="provider-selector">
        <span className="provider-label">Model provider</span>
        <div className="provider-options">
          <label className={`provider-option ${provider === "nvidia" ? "selected" : ""}`}>
            <input
              type="radio"
              name="provider"
              value="nvidia"
              checked={provider === "nvidia"}
              onChange={() => setProvider("nvidia")}
            />
            <span className="provider-name">NVIDIA</span>
            <span className="provider-detail">nemotron-3.5-lightning-30b</span>
          </label>
          <label className={`provider-option ${provider === "ollama" ? "selected" : ""}`}>
            <input
              type="radio"
              name="provider"
              value="ollama"
              checked={provider === "ollama"}
              onChange={() => setProvider("ollama")}
            />
            <span className="provider-name">Ollama</span>
            <span className="provider-detail">llama3.1:8b · local</span>
          </label>
        </div>
      </div>

      <hr />

      <label className="uploader" htmlFor="file-input">
        Upload an invoice or receipt
      </label>
      <input
        id="file-input"
        ref={fileInputRef}
        type="file"
        accept=".pdf,.png,.jpg,.jpeg"
        onChange={handleFileChange}
      />

      {file && (
        <div className="info-box">
          <strong>{file.name}</strong>
          <br />
          Type: <code>{file.type}</code>
          <br />
          Size: {formatSize(file.size)}
        </div>
      )}

      {file && (
        <button className="primary-button" onClick={handleExtract} disabled={loading}>
          {loading
            ? `Reading file, extracting and verifying fields via ${provider === "nvidia" ? "NVIDIA" : "Ollama"}...`
            : "Extract fields"}
        </button>
      )}

      {error && <div className="error-box">{error}</div>}

      {result && (
        <>
          <h2>Extracted fields</h2>
          <p className="caption">Provider used: <strong>{result.provider === "nvidia" ? "NVIDIA" : "Ollama"}</strong></p>
          {result.fields ? (
            <>
              <table className="fields-table">
                <thead>
                  <tr>
                    <th>Field</th>
                    <th>Value</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(result.fields).map(([name, value]) => (
                    <tr key={name}>
                      <td>{name}</td>
                      <td>{value === null ? "null" : String(value)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <button className="secondary-button" onClick={handleDownload}>
                Download JSON
              </button>
            </>
          ) : (
            <div className="error-box">
              <strong>Validation failed:</strong>
              <ul>
                {result.errors.map((err) => (
                  <li key={err}>{err}</li>
                ))}
              </ul>
              <pre>{JSON.stringify(result.raw_json, null, 2)}</pre>
            </div>
          )}

          <details className="text-expander">
            <summary>Extracted document text</summary>
            <pre>{result.raw_text}</pre>
          </details>
        </>
      )}

      {!file && <div className="warning-box">No document uploaded yet. Please upload a PDF, PNG, or JPG to get started.</div>}
    </div>
  );
}
