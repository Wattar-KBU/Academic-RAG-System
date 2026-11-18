import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def clean_text(text: str) -> str:
    """Clean extracted PDF text by removing redundant whitespace and hyphenated line breaks."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'-\s+', '', text)  # Merge words split across line breaks
    return text.strip()

def process_documents(uploaded_files, chunk_size: int = 800, chunk_overlap: int = 150):
    """Load PDF files, sanitize text, and split into semantic chunks while preserving metadata."""
    all_documents = []
    
    for uploaded_file in uploaded_files:
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        loader = PyPDFLoader(temp_path)
        pages = loader.load()
        
        for page in pages:
            page.page_content = clean_text(page.page_content)
            page.metadata["source_file"] = uploaded_file.name
            
        all_documents.extend(pages)
        os.remove(temp_path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    
    chunks = text_splitter.split_documents(all_documents)
    return chunks, len(all_documents)