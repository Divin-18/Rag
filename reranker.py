from sentence_transformers import CrossEncoder
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# 1. Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Connect to Chroma
db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# 3. Load reranker
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


question = "How do I build an API using Python?"


# 4. First stage: Chroma retrieves candidates
results = db.similarity_search(
    question,
    k=5
)


# 5. Prepare question + document pairs
pairs = [
    (question, doc.page_content)
    for doc in results
]


# 6. Reranker scores every pair
scores = reranker.predict(pairs)


# 7. Combine documents + scores
ranked = sorted(
    zip(results, scores),
    key=lambda x: x[1],
    reverse=True
)


# 8. Print final ranking
print("\nRERANKED RESULTS:\n")

for i, (doc, score) in enumerate(ranked, 1):
    print(f"\n[{i}] Score: {score:.4f}")
    print(doc.page_content)