from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from app.pipelines.document_parser import parse_file
from app.services.hybrid_rag import ingest_documents, hybrid_query

app = FastAPI(title="Hybrid RAG Backend API")

class QueryRequest(BaseModel):
    question: str
    run_eval: bool = False

@app.post("/api/ingest")
async def ingest_file(file: UploadFile = File(...)):
    if not file.filename.endswith(('.pdf', '.txt', '.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Unsupported file format.")
    
    content = await file.read()
    docs = parse_file(content, file.filename)
    ingest_documents(docs)
    
    return {"status": "success", "filename": file.filename, "chunks_processed": len(docs)}

@app.post("/api/query")
async def ask_question(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    return hybrid_query(request.question, run_eval=request.run_eval)