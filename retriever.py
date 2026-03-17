# src/retriever.py
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import os

class ResearchRetriever:
    def __init__(self, db_path: str = "data/db"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.collection = self.client.get_or_create_collection(
            name="research_papers",
            metadata={"hnsw:space": "cosine"}
        )

    def add_papers(self, papers: list[dict]):
        """
        Add papers to the vector database.
        Papers should have 'entry_id', 'title', 'summary', and 'authors'.
        """
        documents = [p['summary'] for p in papers]
        metadatas = [{"title": p['title'], "authors": ", ".join(p['authors']), "url": p['pdf_url']} for p in papers]
        ids = [p['entry_id'] for p in papers]
        embeddings = self.model.encode(documents).tolist()

        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def query(self, query_text: str, n_results: int = 5):
        """
        Retrieve relevant papers based on a query.
        """
        query_embedding = self.model.encode([query_text]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        return results
