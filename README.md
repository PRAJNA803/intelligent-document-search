# Intelligent Document Search Engine

An interactive Retrieval-Augmented Generation (RAG) platform that enables semantic search and contextual question answering over uploaded PDF documents using local embeddings and open-source LLMs.

## Features
- **PDF Extraction**: Efficient text parsing and chunking using `pypdf` and `RecursiveCharacterTextSplitter`.
- **Vector Search**: Dense vector embeddings generated via `sentence-transformers` and indexed in `FAISS` for fast similarity matches.
- **Contextual QA**: Powered by `HuggingFacePipeline` with `google/flan-t5-base` for accurate, grounded responses.
- **User Interface**: Built with Streamlit for quick document processing and response inspection.

## Tech Stack
- **Language**: Python
- **Frameworks**: LangChain, Streamlit
- **Vector Store**: FAISS
- **Models**: Hugging Face (`all-MiniLM-L6-v2`, `flan-t5-base`)

## Getting Started

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/intelligent-document-search.git](https://github.com/YOUR_USERNAME/intelligent-document-search.git)
cd intelligent-document-search