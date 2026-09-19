import os
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.app.pipelines.guardrails import guardrails
from backend.app.pipelines.evals import evaluate_rag_response

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def ingest_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(docs)

    Neo4jVector.from_documents(
        chunks,
        embeddings,
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        index_name="vector_index"
    )

    llm_transformer = LLMGraphTransformer(llm=llm)
    graph_docs = llm_transformer.convert_to_graph_documents(chunks)
    graph.add_graph_documents(graph_docs, baseEntityLabel=True, include_source=True)

def hybrid_query(query: str, run_eval: bool = False) -> dict:
    clean_query = guardrails.validate_input(query)

    vector_store = Neo4jVector.from_existing_index(
        embeddings, url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, index_name="vector_index"
    )
    vector_results = vector_store.similarity_search(clean_query, k=3)
    vector_context_chunks = [doc.page_content for doc in vector_results]
    vector_context = "\n".join(vector_context_chunks)

    cypher_query = """
    MATCH (n)-[r]->(m)
    WHERE n.id CONTAINS $query OR m.id CONTAINS $query
    RETURN n.id + ' - ' + type(r) + ' -> ' + m.id AS relationship
    LIMIT 10
    """
    try:
        graph_results = graph.query(cypher_query, params={"query": clean_query})
        graph_context = "\n".join([r["relationship"] for r in graph_results])
    except Exception:
        graph_context = "No direct graph relationships found."

    prompt = f"""
    Answer the user query based on vector text context and knowledge graph relationships.
    Vector Context: {vector_context}
    Graph Context: {graph_context}
    Query: {clean_query}
    """
    raw_answer = llm.invoke(prompt).content
    safe_answer = guardrails.validate_and_scrub_output(raw_answer)

    eval_results = None
    if run_eval:
        eval_results = evaluate_rag_response(
            question=clean_query,
            actual_output=safe_answer,
            retrieval_context=vector_context_chunks + [graph_context]
        )

    return {
        "question": clean_query,
        "answer": safe_answer,
        "evaluations": eval_results
    }