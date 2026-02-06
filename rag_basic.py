from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama

# 1. Load document
loader = TextLoader("data.txt")
documents = loader.load()

# 2. Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)
docs = text_splitter.split_documents(documents)

# 3. Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Store in FAISS
vectorstore = FAISS.from_documents(docs, embeddings)

# 5. Retrieve relevant chunks
retriever = vectorstore.as_retriever()
query = "What is RAG and why is it used?"
relevant_docs = retriever.invoke(query)

context = "\n\n".join([doc.page_content for doc in relevant_docs])

# 6. Load local LLM
llm = Ollama(model="phi")

# 7. Create grounded prompt
prompt = f"""
You are a precise assistant.

Rules:
- Answer ONLY using the context below.
- If the answer is not in the context, say "I don't know".
- Answer in maximum 3 sentences.
- Do NOT add examples, stories, or extra explanations.
- Stop after answering.

Context:
{context}

Question:
{query}

Answer:
"""


# 8. Get answer
result = llm.invoke(prompt)

print("\nAnswer:")
print(result)
