# from langchain_chroma import Chroma
# from langchain_community.document_loaders import TextLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings


# def ingest():
#     loader = TextLoader("sample.txt")
#     docs = loader.load()
#     splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
#     chunks = splitter.split_documents(docs)

#     for i, chunk in enumerate(chunks):
#         text = chunk.page_content.lower()

#         if any(word in text for word in ["fastapi", "python", "django"]):
#             category = "backend"
#         elif any(word in text for word in ["postgresql", "redis"]):
#             category = "database"
#         elif any(word in text for word in ["chroma", "embeddings", "rag"]):
#             category = "rag"
#         elif any(word in text for word in ["docker", "kubernetes"]):
#             category = "devops"
#         elif any(word in text for word in ["react", "typescript"]):
#             category = "frontend"
#         else:
#             category = "other"

#         chunk.metadata["source"] = "sample.txt"
#         chunk.metadata["document_id"] = "doc-001"
#         chunk.metadata["category"] = category
#         chunk.metadata["chunk_id"] = i

#     print(f"Generated {len(chunks)} chunks:")
#     for i, c in enumerate(chunks):
#         print(f"  [{i}] {c.page_content[:60]}...")

#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#     db = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
#     print(f"✅ Stored {len(chunks)} chunks in ./chroma_db")


# if __name__ == "__main__":
#     ingest()

from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


def ingest():

    # --------------------------------------------------------
    # 1. Find PDF files
    # --------------------------------------------------------

    pdf_files = list(Path("documents").glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in ./documents")
        return

    print(f"Found {len(pdf_files)} PDF(s)")


    # --------------------------------------------------------
    # 2. Load PDFs
    # --------------------------------------------------------

    all_docs = []

    for pdf_file in pdf_files:

        print(f"Loading: {pdf_file.name}")

        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()

        all_docs.extend(docs)


    # --------------------------------------------------------
    # 3. Split documents into chunks
    # --------------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(all_docs)

    print(f"Generated {len(chunks)} chunks")


    # --------------------------------------------------------
    # 4. Add metadata
    # --------------------------------------------------------

    for i, chunk in enumerate(chunks):

        source_path = Path(chunk.metadata["source"])
        filename = source_path.name

        chunk.metadata["document_id"] = source_path.stem
        chunk.metadata["chunk_id"] = i
        chunk.metadata["source"] = filename

        # Calculate the lowercase stem once
        stem_lower = source_path.stem.lower()

        if "finance" in stem_lower:
            category = "finance"
        elif "engineering" in stem_lower:
            category = "backend"
        else:
            category = "other"

        chunk.metadata["category"] = category


    # --------------------------------------------------------
    # 5. Embeddings
    # --------------------------------------------------------

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


    # --------------------------------------------------------
    # 6. Store in Chroma
    # --------------------------------------------------------

    Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory="./chroma_db"
    )

    print(f"Stored {len(chunks)} chunks in ./chroma_db")


    # --------------------------------------------------------
    # 7. Show sample chunks
    # --------------------------------------------------------

    print("\nSample chunks:\n")

    for i, chunk in enumerate(chunks[:5]):

        print(f"[{i}]")
        print(chunk.page_content[:300])
        print("Metadata:", chunk.metadata)
        print()


if __name__ == "__main__":
    ingest()