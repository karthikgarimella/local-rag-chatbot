# Local RAG PDF Chatbot
A fully local, privacy-first PDF Question & Answer chatbot using Retrieval-Augmented Generation (RAG). Answers questions strictly from uploaded documents without using any paid APIs or cloud services.

🎯 Overview
This project demonstrates practical RAG implementation for document-based Q&A. The system runs entirely on your local machine, preserves data privacy, and reduces hallucinations by grounding answers in retrieved context.
Key Benefits:

✅ Fully local execution (no cloud, no APIs)
✅ Complete data privacy
✅ Zero cost (open-source models)
✅ Answers only from provided documents


🏗️ How It Works
PDF Upload → Text Extraction → Chunking → Embeddings → 
FAISS Vector Store → Semantic Search → LLM Answer Generation

PDF text is extracted and split into chunks
Chunks are embedded using sentence-transformers
FAISS indexes embeddings for similarity search
User questions retrieve relevant chunks
Local LLM generates answers using only retrieved context


🧰 Tech Stack
ComponentTechnologyUIStreamlitRAG FrameworkLangChain CommunityVector DBFAISSEmbeddingssentence-transformers (MiniLM-L6-v2)LLMOllama (phi model)

🚀 Quick Start
Prerequisites

Python 3.10+
Ollama installed

Installation
bash# Clone repository
git clone https://github.com/karthikgarimella/local-rag-chatbot.git
cd local-rag-chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install streamlit langchain langchain-community langchain-text-splitters \
    sentence-transformers faiss-cpu ollama

# Install and start Ollama
ollama pull phi
ollama serve
Run
streamlit run ch_code.py
Open browser at http://localhost:8501

💡 Usage

Upload a text-based PDF
Ask questions in natural language
Get answers grounded in document content

Example Questions:

"What is the main topic of this document?"
"Summarize the key findings"
"What requirements are mentioned?"


⚠️ Limitations
Supported:
✅ Text-based PDFs (reports, papers, documentation)
Not Supported:
❌ Scanned PDFs or image-based documents (OCR can be added as future enhancement)
Performance:

Response time slower than cloud LLMs (local CPU inference)
Best for documents under 100 pages


🔧 Technical Highlights
Chunking Strategy:

500 characters per chunk
100-character overlap (20%)
Hierarchical separators for better context

Retrieval:

Top-4 most relevant chunks
Cosine similarity search
384-dimensional embeddings

Design Decisions:

Custom implementation for full control
FAISS for local-first approach
Explicit caching with file hash
Proper error handling and cleanup


🎯 Use Cases

Privacy-sensitive document analysis
Offline/air-gapped environments
Internal knowledge base search
Learning RAG fundamentals
Cost-conscious applications


🔮 Future Improvements

 Conversation history/memory
 Multi-PDF support
 Page number citations
 OCR for scanned documents
 Evaluation metrics (RAGAS)


📄 License
MIT License - Free to use and modify

🙏 Acknowledgments
Built with: Ollama • LangChain • FAISS • Sentence Transformers • Hugging Face

⭐ Star this repo if you find it helpful!

