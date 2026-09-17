from fastapi import FastAPI
from pydantic import BaseModel

# RAG / Vector database
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Local LLM
from langchain_ollama import ChatOllama

# Reranker
from sentence_transformers import CrossEncoder

# OLD: No longer needed because we manually handle retrieval + reranking
# from langchain_core.prompts import PromptTemplate
# from langchain_classic.chains import RetrievalQA


app = FastAPI(title="Simple RAG API")


# ============================================================
# 1. EMBEDDING MODEL
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# 2. VECTOR DATABASE
# ============================================================

db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)


# ============================================================
# 3. RERANKER
# ============================================================

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# ============================================================
# 4. LOCAL LLM
# ============================================================

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)


# ============================================================
# OLD RAG PROMPT
# ============================================================

# We don't need PromptTemplate now because we create the
# prompt directly inside /query using an f-string.

# template = """Use ONLY the following context to answer the question.
# Do NOT make assumptions, do NOT mention tricks or games,
# and do NOT add commentary.
# If the answer is not in the context, say
# "I don't know based on the provided information."
#
# Context: {context}
#
# Question: {question}
#
# Answer:"""

# PROMPT = PromptTemplate(
#     template=template,
#     input_variables=["context", "question"]
# )


# ============================================================
# OLD RETRIEVAL QA CHAIN
# ============================================================

# We no longer use RetrievalQA because we are manually doing:
#
# Chroma retrieval
#       ↓
# Reranking
#       ↓
# Top documents
#       ↓
# LLM
#
# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     retriever=db.as_retriever(
#         search_kwargs={"k": 2}
#     ),
#     return_source_documents=True,
#     chain_type_kwargs={"prompt": PROMPT}
# )


# ============================================================
# REQUEST MODEL
# ============================================================

class QueryRequest(BaseModel):
    question: str
    category: str | None = None


# ============================================================
# OLD /query ENDPOINT
# ============================================================

# This was the original RAG flow:
#
# Chroma
#   ↓
# Top 2
#   ↓
# LLM
#
# @app.post("/query")
# async def query(req: QueryRequest):
#     result = qa_chain.invoke({
#         "query": req.question
#     })
#
#     return {
#         "answer": result["result"],
#         "sources": [
#             doc.page_content
#             for doc in result["source_documents"]
#         ]
#     }


# ============================================================
# OLD /search ENDPOINT
# ============================================================

# @app.post("/search")
# async def search(req: QueryRequest):
#
#     results = db.similarity_search_with_score(
#         req.question,
#         k=5
#     )
#
#     return {
#         "question": req.question,
#         "results": [
#             {
#                 "content": doc.page_content,
#                 "score": score
#             }
#             for doc, score in results
#         ]
#     }


# ============================================================
# Rewriting Query
# ============================================================

def rewrite_query(question: str) -> str:
    prompt = f"""
Rewrite the user's question into a concise search query
for retrieving relevant information from a knowledge base.

Keep the important technical terms.
Do not answer the question.
Return only the rewritten search query.

User question:
{question}

Search query:
"""

    response = llm.invoke(prompt)

    return response.content.strip()

# ============================================================
# NEW RERANKED RAG ENDPOINT
# ============================================================   

@app.post("/query")
async def query(req: QueryRequest):

    # --------------------------------------------------------
    # STEP 1: Retrieve candidates from Chroma
    # --------------------------------------------------------
    search_query = rewrite_query(req.question)

    print(f"Original query: {req.question}")
    print(f"Rewritten query: {search_query}")

    if req.category:

        documents = db.similarity_search(
            search_query,
            k=5,
            filter={
                "category": req.category
            }
        )

    else:

        documents = db.similarity_search(
            search_query,
            k=5
        )


    print("Retrieved documents:", len(documents))


    # --------------------------------------------------------
    # STEP 2: Handle no documents
    # --------------------------------------------------------

    if not documents:

        return {
            "answer": "I don't know based on the provided information.",
            "sources": []
        }


    # --------------------------------------------------------
    # STEP 3: Create question-document pairs
    # --------------------------------------------------------

    pairs = [
        (req.question, doc.page_content)
        for doc in documents
    ]


    # --------------------------------------------------------
    # STEP 4: Reranker scores each pair
    # --------------------------------------------------------

    scores = reranker.predict(pairs)


    # --------------------------------------------------------
    # STEP 5: Sort documents by reranker score
    # Higher score = more relevant
    # --------------------------------------------------------

    ranked = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True
    )


    # --------------------------------------------------------
    # TEMP: Print reranker scores for testing
    # --------------------------------------------------------

    for doc, score in ranked:

        print(
            f"Score: {score:.4f} | "
            f"{doc.page_content}"
        )


    # --------------------------------------------------------
    # STEP 6: Check relevance threshold
    # --------------------------------------------------------

    best_score = ranked[0][1]

    print("Best score:", best_score)


    if best_score < -9:

        return {
            "answer": "I don't know based on the provided information.",
            "best_score": float(best_score),
            "sources": []
        }


    # --------------------------------------------------------
    # STEP 7: Keep only the best 2 documents
    # --------------------------------------------------------

    top_documents = [
        doc
        for doc, score in ranked[:2]
    ]


    # --------------------------------------------------------
    # STEP 8: Build context for the LLM
    # --------------------------------------------------------

    context = "\n\n".join(
        doc.page_content
        for doc in top_documents
    )


    # --------------------------------------------------------
    # STEP 9: Create the LLM prompt
    # --------------------------------------------------------

    prompt = f"""
You are a question-answering assistant.

Use the provided context to answer the question.

Rules:
- Answer using information from the context.
- Do not use outside knowledge.
- If the context does not contain the answer, say:
  "I don't know based on the provided information."
- Keep the answer concise.

Context:
{context}

Question:
{req.question}

Answer:
"""


    # --------------------------------------------------------
    # STEP 10: Send the reranked context to Llama
    # --------------------------------------------------------

    response = llm.invoke(prompt)


    # --------------------------------------------------------
    # STEP 11: Return answer + sources
    # --------------------------------------------------------

    return {
        "answer": response.content,
        "best_score": float(best_score),
        "sources": [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in top_documents
        ]
    }