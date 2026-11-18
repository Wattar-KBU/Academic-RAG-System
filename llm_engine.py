import google.generativeai as genai

def generate_grounded_response(api_key: str, query: str, retrieved_docs: list) -> str:
    if not retrieved_docs:
        return "Information not found in the uploaded regulations."

    formatted_context = ""
    for i, doc in enumerate(retrieved_docs, 1):
        source = doc.metadata.get("source_file", "Unknown")
        page = doc.metadata.get("page_number", "N/A")
        formatted_context += f"--- Chunk {i} (Source: {source}, Page: {page}) ---\n{doc.page_content}\n\n"

    # تهيئة مفتاح الـ API باستخدام مكتبة جوجل الرسمية
    genai.configure(api_key=api_key)
    
    # استخدام النموذج الافتراضي المدعوم تلقائياً
    model = genai.GenerativeModel('gemini-1.5-flash')

    prompt = f"""You are an Academic and Legal Document Intelligence Assistant.
Answer strictly based on the provided context with citations (mentioning Document Name and Page Number).

Context:
{formatted_context}

Question: {query}
"""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        # كخيار بديل تلقائي إذا تطلب الأمر نموذجاً آخر
        try:
            model_alt = genai.GenerativeModel('gemini-pro')
            response = model_alt.generate_content(prompt)
            return response.text
        except Exception as e2:
            return f"Error generating response: {str(e)}"