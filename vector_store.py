import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

INDEX_DIR = "faiss_academic_index"

def get_embedding_model():
    """Load the multilingual dense embedding model (supports English and Turkish)."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

def build_and_save_index(chunks):
    """Build a FAISS vector index from document chunks and persist it locally on disk."""
    embeddings = get_embedding_model()
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local(INDEX_DIR)
    return vector_db

def load_existing_index():
    """Load the persisted FAISS index from local storage if available."""
    if os.path.exists(INDEX_DIR):
        embeddings = get_embedding_model()
        return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    return None

def retrieve_relevant_chunks(vector_db, query: str, k: int = 4, use_mmr: bool = True):
    """Retrieve the Top-K most relevant chunks using either MMR (diversity) or Cosine Similarity."""
    if use_mmr:
        retriever = vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": k * 3}
        )
    else:
        retriever = vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
    return retriever.invoke(query)