# 🛒 Wise Cart Assistant

🔗 [HuggingFace Project Repository](#)  
🔗 [Amazon Dataset CSV](#)  
🔗 [Live Gradio Demo](#)

## 👩‍💻 Team Members
Hager Hamed  
Ahmed Elshamy  

Under Supervision:  
Eng. Mohammed Agoor  
Instructor of Data Science & AI in Digital Egypt Pioneers Initiative  

## 📝 Project Description
Wise Cart Assistant is an AI-powered shopping assistant that helps users discover, compare, and choose products based on their needs, budget, and preferences.  

Key features include:  
- Product recommendations using RAG (Retrieval-Augmented Generation).  
- Explains why each recommended product fits user requirements.  
- Provides comparison on price, features, and value.  
- Suggests reliable places to buy and payment methods.  
- Friendly and natural conversation handling for casual messages.  

This project addresses challenges like:  
- Handling large datasets of Amazon products efficiently.  
- Integrating FAISS vector search with embeddings for fast retrieval.  
- Leveraging LLMs (Groq LLaMA-3.3-70B) for context-aware product suggestions.  
- Maintaining chat history for consistent user interactions.  

## 🛠 Techniques Used
- Groq LLaMA-3.3-70B model  
- RAG (Retrieval-Augmented Generation)  
- FAISS vector database  
- HuggingFace embeddings (sentence-transformers/all-MiniLM-L6-v2)  
- Gradio for web UI deployment  
- Pandas for data handling  

## 🚀 API Overview
- Retrieve product info from the Amazon dataset.  
- Generate recommendations and comparisons dynamically.  
- Maintain friendly chat experience with users.  

## 🔥 Requirements
```bash
pip install -r requirements.txt
