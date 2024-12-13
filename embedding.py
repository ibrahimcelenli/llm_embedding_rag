from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import pandas as pd
import os

# --- Embedding Modeli ve VectorStore ---
df = pd.read_csv("netflix_titles.csv").fillna("Unknown")

del df["show_id"]
del df["cast"]
del df["rating"] 
film_data = df.copy() 

# Generating documents and checking for missing values
documents = [
    Document(
        page_content=(
            f"Title: {film['title']}\n"
            f"Type: {film['type']}\n"
            f"Director: {film['director']}\n"
            f"Country: {film['country']}\n"
            f"Year: {film['release_year']}\n"
            f"Duration: {film['duration']}\n"
            f"Genre: {film['listed_in']}\n"
            f"Description: {film['description']}"
        ),
        metadata={
            "title": film["title"],
            "year": film["release_year"],
            "type": film["type"]
        }
    )
    for _, film in film_data.iterrows()
]

# HuggingFace Embedding Model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ChromaDB'ye Belgeleri Kaydetme
persist_directory = "chroma_db"
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

vectorstore = Chroma.from_documents(documents, embeddings, persist_directory=persist_directory)

print("The embedding process is complete and the data is saved to ChromaDB.")

