import os
import pandas as pd
from io import BytesIO
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

def parse_file(file_bytes: bytes, filename: str) -> list[Document]:
    docs = []
    
    if filename.endswith(".pdf"):
        temp_path = f"C:\\Windows\\Temp\\{filename}" if os.name == "nt" else f"/tmp/{filename}"
        with open(temp_path, "wb") as f:
            f.write(file_bytes)
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        if os.path.exists(temp_path):
            os.remove(temp_path)

    elif filename.endswith(".txt"):
        text = file_bytes.decode("utf-8")
        docs = [Document(page_content=text, metadata={"source": filename})]

    elif filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(BytesIO(file_bytes))
        text_data = ""
        for idx, row in df.iterrows():
            row_str = ", ".join([f"{col}: {val}" for col, val in row.items()])
            text_data += f"Row {idx+1}: {row_str}\n"
        docs = [Document(page_content=text_data, metadata={"source": filename})]

    return docs