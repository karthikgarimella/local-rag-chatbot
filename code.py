import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama

st.set_page_config(page_title="Local RAG PDF Chatbot", layout="centered")
st.title("📄 Local RAG PDF Chatbot")

st.write("Upload a PDF and ask questions. The chatbot answers only from the document.")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

@st.cache_resource
def build_rag(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    llm = Ollama(model="phi", temperature=0)
    return retriever, llm

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    retriever, llm = build_rag("temp.pdf")

    query = st.text_input("Ask a question from the PDF")

    if query:
        docs = retriever.invoke(query)
        context = "\n\n".join([d.page_content for d in docs])

        prompt = f"""
Answer using ONLY the context below.
If the answer is not in the context, say "I don't know".
Answer in at most 2 sentences.

Context:
{context}

Question:
{query}

Answer:
"""

        with st.spinner("Thinking..."):
            response = llm.invoke(prompt)

        st.markdown("### 💬 Answer")
        st.write(response)
