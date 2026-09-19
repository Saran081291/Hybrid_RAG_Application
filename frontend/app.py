import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Hybrid RAG App", layout="wide")
st.title("⚡ Hybrid RAG System (Vector + Knowledge Graph)")

# Sidebar for Document Ingestion
with st.sidebar:
    st.header("1. Document Ingestion")
    uploaded_file = st.file_uploader("Upload PDF, TXT, or Excel", type=["pdf", "txt", "xlsx", "xls"])
    
    if st.button("Process & Index") and uploaded_file is not None:
        with st.spinner("Processing document..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            try:
                res = requests.post(f"{API_URL}/api/ingest", files=files)
                if res.status_code == 200:
                    st.success("Document processed and indexed successfully!")
                else:
                    try:
                        err_msg = res.json().get("detail", res.text)
                    except Exception:
                        err_msg = res.text
                    st.error(f"Error ({res.status_code}): {err_msg}")
            except Exception as e:
                st.error(f"Failed to connect to backend: {e}")

    run_evals = st.checkbox("Run DeepEval Metrics on Query", value=False)

# Main Chat Interface
st.header("2. Ask Questions")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching Knowledge Graph & Vector Store..."):
            try:
                payload = {"question": prompt, "eval_flag": run_evals}
                res = requests.post(f"{API_URL}/api/query", json=payload)
                
                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("answer", "No response content received.")
                    st.markdown(answer)
                    
                    if "eval_scores" in data and data["eval_scores"]:
                        st.json(data["eval_scores"])
                        
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    try:
                        err_msg = res.json().get("detail", res.text)
                    except Exception:
                        err_msg = res.text
                    st.error(f"Error ({res.status_code}): {err_msg}")
            except Exception as e:
                st.error(f"Failed to reach backend: {e}")