# llm_embedding_rag
Multilingual movie search engine based on LLM and Embedding

The aim was to prevent LLM from responding to everything and to provide feedback to users with embeding and accessible data.

![Example](images/film_bot_screenshot.png)

## 🚀 **Features**

- **Multilingual Support**: The bot responds in the same language as the user's input.
- **Accurate Movie Information**: Utilizes **ChromaDB** for semantic search and returns relevant movie data.
- **Casual Interaction**: Handles greetings like *"hello"*, *"selam"* with friendly responses.
- **Controlled Responses**: If no relevant movie data is found, the bot avoids generating hallucinated answers.

---

## 🛠️ **Technologies Used**

- **Python**: Core programming language.
- **LangChain**: Framework for building LLM-powered applications.
- **ChromaDB**: Vector database for semantic search.
- **HuggingFace Sentence Transformers**: Embedding model for converting queries into vector representations.
- **OpenAI GPT-4o**: Large Language Model for natural language generation.
- **Streamlit**: Interactive web interface for user input and chatbot responses.

---
