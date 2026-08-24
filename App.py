import streamlit as st

from backend import extract_invoice

st.set_page_config(page_title="Invoice / Receipt Field Extractor", page_icon="🧾", layout="centered")

st.title("🧾 Invoice / Receipt Field Extractor")
st.caption(
    "Upload in the browser, one prompt extracts the fields, Pydantic enforces every "
    "format, and the result renders on screen."
)

st.divider()

uploaded_file = st.file_uploader("Upload an invoice or receipt", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file:
    size_kb = len(uploaded_file.getvalue()) / 1024
    st.info(f"**{uploaded_file.name}**  \nType: `{uploaded_file.type}`  \nSize: {size_kb:.1f} KB")

    if st.button("Extract fields"):
        try:
            with st.spinner("Reading file, extracting and verifying fields..."):
                result = extract_invoice(uploaded_file)
        except Exception as exc:
            st.error(str(exc))
        else:
            st.subheader("Extracted fields")
            if result.fields:
                rows = [
                    {"field": name, "value": "null — blank, never guessed" if value is None else value}
                    for name, value in result.fields.model_dump(mode="json").items()
                ]
                st.dataframe(rows, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download JSON",
                    data=result.fields.model_dump_json(indent=2),
                    file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}.json",
                    mime="application/json",
                )
            else:
                st.error("Validation failed:")
                for err in result.errors:
                    st.write(f"- {err}")
                st.json(result.raw_json)

            with st.expander("Extracted document text"):
                st.text(result.raw_text)
else:
    st.warning("No document uploaded yet. Please upload a PDF, PNG, or JPG to get started.")
