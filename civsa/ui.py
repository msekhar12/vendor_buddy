"""Minimal Streamlit UI covering upload + Q&A."""
import requests
import streamlit as st

from civsa.config import ALLOWED_LABELS

API = "http://localhost:8000"

st.set_page_config(page_title="CIVSA · Phase 1 demo", layout="wide")
tab_upload, tab_qa = st.tabs(["Upload", "Ask"])

# ---------------- Upload tab ----------------
with tab_upload:
    st.header("Upload vendor documents")
    vendor = st.text_input("Vendor name")
    files = st.file_uploader("Documents", accept_multiple_files=True,
                             type=["pdf", "docx", "xlsx", "txt"])
    label_options = sorted(ALLOWED_LABELS)

    if files and vendor:
        for f in files:
            with st.expander(f"{f.name}  ({len(f.getvalue())//1024} KB)",
                             expanded=True):
                labels = st.multiselect(
                    "Labels", label_options, default=["quote"], key=f.name)
                if st.button("Upload", key=f"up_{f.name}"):
                    r = requests.post(
                        f"{API}/api/upload",
                        data={"vendor": vendor, "labels": labels,
                              "uploaded_by": "priya@acme.in"},
                        files={"file": (f.name, f.getvalue())})
                    body = r.json() if r.ok else {"error": r.text}
                    if body.get("status") == "duplicate":
                        st.warning(f"Duplicate of {body['linked_to']}")
                    elif body.get("status") == "stored":
                        st.success(f"Stored: {body['path']}")
                    else:
                        st.error(body)

# ---------------- Q&A tab ----------------
with tab_qa:
    st.header("Ask a question")
    q = st.text_input(
        "Your question",
        placeholder="e.g., Which vendors have ISO 9001?")
    if st.button("Ask") and q:
        r = requests.post(f"{API}/api/query",
                          data={"q": q, "user": "priya"})
        body = r.json()
        if not body["allowed"]:
            st.warning(body["message"])
        else:
            st.info(f"Routed to intent: **{body['intent']}**")
            st.write(body["answer"])
            with st.expander("Sources"):
                st.json(body.get("sources", []))