
# ChatGroq RAG Assistant
A Document-QA system powered by Groq, FAISS & LangChain.

# 📘 Description
ChatGroq RAG Assistant is an intelligent Document Question-Answering (RAG) application that allows users to query PDFs using natural language.

The system combines:

Groq’s Llama-3.1 LLM for ultra-fast responses OpenAI embeddings to represent text semantically FAISS for high-performance similarity search. LangChain LCEL to build a clean, maintainable RAG pipeline. Streamlit for a lightweight user interface. Users can upload PDF documents, generate embeddings, and interact with the content conversationally.The system retrieves the most relevant chunks and passes them to the LLM, producing accurate answers grounded in the actual documents.
## Features

- PDF Ingestion : 
Loads PDF files using PyPDFDirectoryLoader. Splits documents into manageable chunks using RecursiveCharacterTextSplitter.
- Embedding & Vector Store Creation :
Generates embeddings with OpenAIEmbeddings. Builds a FAISS vector database from the processed documents. Stores embeddings and vector database in st.session_state for reuse.
- Robust Retrieval-Augmented Generation (RAG) :
Uses a safe LCEL retrieval chain: Retriever → Formatter → Prompt → Groq LLM → Output Parser. 

Includes a custom retriever wrapper ensuring: Only string queries reach the vector store Prevents common LCEL dict-related errors Always returns a clean list of Document objects.

- Powerful Groq-Driven Answering :
Uses ChatGroq (Llama-3.1-8b-instant) to generate fast, high-quality answers.
Keeps the responses grounded in the retrieved document context.

- Clean, Structured Output :
Responses are formatted into a nice, flattened dictionary that works seamlessly with Streamlit’s st.write().
~~~
{
  "question": "...",
  "tool_used": "...",
  "tool_result": "...",
  "final_answer": "..."
}
~~~

- Document Similarity Search Viewer :
If the LCEL pipeline does not return the context, we fallback to the FAISS retriever directly.
Displays: Relevant chunks retrieved Source file (if available) Clean extracted text

- Streamlit UI :
Button to trigger document indexing, Input box for user questions, Formatted model outputs & Expandable section to inspect retrieved documents.




## 🛠️Tech Stack


**Frontend** - 	Streamlit

**LLM	Groq** - (Llama-3.1-8b-instant)

**Embeddings** - 	OpenAIEmbeddings

**Vector Store** -	FAISS

**RAG Pipeline** -	LangChain + LCEL

**Environment Management** - Python + dotenv
