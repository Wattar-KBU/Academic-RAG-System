import streamlit as st
from ingestion import process_documents
from vector_store import build_and_save_index, load_existing_index, retrieve_relevant_chunks
from llm_engine import generate_grounded_response

st.set_page_config(page_title="Academic RAG Intelligence", page_icon="⚖️", layout="wide")
st.title("⚖️ Academic & Legal Document Intelligence System")
st.caption("Retrieval-Augmented Generation (RAG) Decision Support System")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for system hyperparameters and document ingestion
with st.sidebar:
    st.header("🛠️ System Parameters")
    api_key = st.text_input("Gemini API Key:", type="password")
    
    st.subheader("1. Chunking Settings")
    chunk_size = st.slider("Chunk Size (Characters)", 300, 1500, 800, step=100)
    chunk_overlap = st.slider("Chunk Overlap", 50, 300, 150, step=25)
    
    st.subheader("2. Retrieval Settings")
    top_k = st.slider("Top-K Retrieved Chunks", 2, 8, 4)
    use_mmr = st.checkbox("Use MMR (Diversity Search)", value=True)
    
    st.divider()
    st.subheader("3. Knowledge Base")
    uploaded_files = st.file_uploader("Upload Regulations (PDF)", type=["pdf"], accept_multiple_files=True)
    
    if st.button("🚀 Build / Update Vector Index", use_container_width=True):
        if uploaded_files:
            with st.spinner("Cleaning, Chunking & Indexing..."):
                chunks, total_pages = process_documents(uploaded_files, chunk_size, chunk_overlap)
                build_and_save_index(chunks)
                st.success(f"Indexed {len(chunks)} chunks from {total_pages} pages!")
        else:
            st.warning("Please upload at least one PDF file.")

# Load persisted vector database if available
vector_db = load_existing_index()

if not vector_db:
    st.info("👈 Upload your university regulations in the sidebar and click **Build / Update Vector Index** to initialize the system.")
else:
    # Render previous chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Handle new user query
    if prompt := st.chat_input("Ask a question about university regulations..."):
        if not api_key:
            st.error("Please enter your Gemini API Key in the sidebar.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Searching regulations & verifying citations..."):
                    # Step 1: Retrieve relevant chunks (Student 2)
                    docs = retrieve_relevant_chunks(vector_db, prompt, k=top_k, use_mmr=use_mmr)
                    # Step 2: Generate grounded answer (Student 3)
                    answer = generate_grounded_response(api_key, prompt, docs)
                    
                    st.markdown(answer)
                    
                    # Display retrieved context and source citations
                    with st.expander("🔍 Inspect Retrieved Context & Citations"):
                        for i, doc in enumerate(docs, 1):
                            st.markdown(f"**Chunk {i}** | File: `{doc.metadata.get('source_file')}` | Page: `{doc.metadata.get('page', 0) + 1}`")
                            st.info(doc.page_content)
                            
            st.session_state.messages.append({"role": "assistant", "content": answer})