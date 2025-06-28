# Assignment 3.1: Retrieval-Augmented Generation (RAG)

This assignment demonstrates a basic implementation of a Retrieval-Augmented Generation (RAG) system. It involves crawling a target website, constructing a retrieval index, and generating answers using context retrieved from the document store and a language model.

## 🎯 Goals
- Crawl web content and build a structured document store  
- Implement a retriever to locate relevant content for user queries  
- Use a language model to generate responses based on retrieved information  
- Evaluate the end-to-end RAG pipeline using a QA dataset like Natural Questions or TriviaQA

## 🚀 How to Use
1. Install all dependencies: `pip install -r requirements.txt`  
2. Open and run the notebook: `rag_pipeline.ipynb`  
3. Place any datasets or crawled documents in the `data/` directory

## 📁 Project Structure
- `rag_pipeline.ipynb` – Core implementation of the RAG pipeline  
- `requirements.txt` – List of required Python packages  
- `data/` – Folder containing datasets and crawled web content  
- `utils.py` – Utility module with helper functions for crawling, retrieval, and evaluation

---
