import streamlit as st
import tempfile
import hashlib
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama


st.set_page_config(page_title="Local RAG Chatbot", layout="wide")
st.title("📄 Local PDF Q&A Chatbot (RAG)")


@st.cache_resource
def build_rag(pdf_path: str, file_hash: str):
    # Loading PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Splittign text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # Local LLM 
    llm = Ollama(
        model="phi",
        temperature=0,
        num_predict=120
    )

    return retriever, llm


# File upload

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:

    file_bytes = uploaded_file.read()
    file_hash = hashlib.md5(file_bytes).hexdigest()

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_file.write(file_bytes)
    temp_file.close()

    with st.spinner("📚 Processing PDF and building knowledge base..."):
        try:
            retriever, llm = build_rag(temp_file.name, file_hash)
            st.success("✅ PDF processed successfully")
        except Exception as e:
            st.error("❌ Failed to process PDF")
            st.stop()

    # Cleanup temp file
    os.unlink(temp_file.name)

    query = st.text_input("Ask a question about the PDF")

    if query:
        # Retrieve documents
        docs = retriever.invoke(query)
        context = "\n\n".join([d.page_content for d in docs])

        # Prompt
        prompt = f"""
You are a helpful assistant.

RULES:
- Answer ONLY using the context below
- If the answer is not in the context, say "I don't have enough information"
- Keep the answer short (2–3 sentences)

Context:
{context}

Question:
{query}

Answer:
"""

        st.subheader("Answer")

        answer_placeholder = st.empty()
        full_response = ""

        try:
            for chunk in llm.stream(prompt):
                full_response += chunk
                answer_placeholder.markdown(full_response)
        except Exception:
            st.error("❌ Ollama is not running. Please start Ollama.")

