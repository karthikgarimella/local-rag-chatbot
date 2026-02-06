Local Q&A Chatbot using RAG
🔍 Project Overview

This project is a fully local Retrieval-Augmented Generation (RAG) based Question & Answer chatbot that allows users to upload a PDF document and ask questions about its content.

Unlike cloud-based solutions, this chatbot:

Runs entirely on a local machine

Uses open-source LLMs

Does not rely on any paid APIs

Answers strictly from the uploaded document, reducing hallucinations

The goal of this project is to demonstrate a clear, practical understanding of RAG architecture, vector search, and LLM grounding — not just tool usage.

🧠 What Problem Does This Solve?

Large Language Models (LLMs) often:

Hallucinate answers

Lack access to private or domain-specific data

Require expensive cloud APIs

This project solves that by:

Retrieving relevant document chunks using embeddings + vector search

Injecting those chunks into the LLM prompt

Forcing the model to answer only from retrieved context

🏗️ Architecture (RAG Flow)
User Question
   ↓
Convert question to embedding
   ↓
FAISS Vector Search (Top-K relevant chunks)
   ↓
Context Injection into Prompt
   ↓
Local LLM (Ollama)
   ↓
Grounded Answer


This explicit pipeline avoids high-level black-box abstractions and keeps the system transparent and explainable.

🧰 Tech Stack & Rationale
Component	Technology	Why It Was Used
LLM	Ollama (phi / tinyllama)	Fully local inference, no paid APIs
Embeddings	sentence-transformers (MiniLM)	Lightweight and fast on CPU
Vector DB	FAISS	Efficient similarity search
RAG Logic	Custom retrieval + prompt injection	Avoids unstable high-level wrappers
UI	Streamlit	Simple, fast chatbot interface
Language	Python	Strong ecosystem for GenAI
🚀 Features

📤 Upload any PDF document

💬 Ask natural language questions

🧠 Retrieval-Augmented Generation (true RAG)

🚫 No hallucinated answers (context enforced)

🖥️ Fully local execution (privacy-first)

⚡ Optimized for CPU-only systems

▶️ How to Run Locally
1️⃣ Prerequisites

Python 3.10+

Ollama installed and running

Local LLM pulled (e.g. phi or tinyllama)

ollama pull phi

2️⃣ Clone / Download Project

Place all files inside a folder, e.g.:

rag_chatbot/
├── app.py
├── rag_basic.py
├── README.md

3️⃣ Create & Activate Virtual Environment
py -m venv venv
venv\Scripts\activate

4️⃣ Install Dependencies
pip install streamlit langchain-community langchain-text-splitters \
           sentence-transformers faiss-cpu pypdf

5️⃣ Run the Chatbot
streamlit run app.py


Open the browser at:

http://localhost:8501


Upload a PDF and start asking questions.

🧠 Key Design Decisions (Interview Focus)
Why Local LLM instead of OpenAI / GPT?

Avoids cost and rate limits

Preserves data privacy

Demonstrates real understanding of model constraints

Makes the system reproducible offline

Why Small Models Work Well Here

Because RAG supplies relevant context, the LLM:

Does not need to “remember” everything

Only needs to reason over retrieved text

This allows smaller models to perform well with lower resource usage.

How Hallucinations Are Reduced

Answers are restricted to retrieved document chunks

Prompt explicitly forbids external knowledge

If information is missing, the model responds with “I don’t know”

Why Not Use High-Level LangChain Wrappers?

LangChain APIs change frequently

Explicit retrieval + prompt injection is:

More stable

Easier to debug

Easier to explain in interviews

⚠️ Limitations

Response time is slower than cloud LLMs (CPU-only local inference)

Embeddings are rebuilt when a new PDF is uploaded

Designed for demo and learning, not production scale

🔮 Future Improvements

Persistent vector store per document

Chat history / conversational memory

Support for multiple PDFs

Optional cloud deployment

UI improvements and response streaming

🎯 Intended Audience

This project is aimed at:

GenAI Engineer roles

ML / AI Engineer roles

Candidates learning RAG and LLM systems

Interview demonstrations and technical discussions

🧾 Summary

This project demonstrates:

Practical RAG implementation

Clear understanding of GenAI system design

Ability to work with local LLMs and real constraints

Strong debugging and architectural reasoning

📌 Note

Response latency is expected due to fully local execution on CPU.
This is an intentional trade-off for cost, privacy, and transparency.