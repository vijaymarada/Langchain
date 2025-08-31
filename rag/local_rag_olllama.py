import chromadb
from chromadb import PersistentClient
from chromadb.config import Settings
import requests
import json
import numpy as np

chroma_client = PersistentClient(path="./chroma_data")

# create and populate the collection
collection = chroma_client.get_or_create_collection("my_collection")

def get_ollama_embedding(text):
    url = "http://localhost:11434/v1/embeddings"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "llama3",
        "input": text
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        embedding = response.json().get("data", [])[0].get("embedding", [])
        return np.array(embedding)
    return None

def add_document(doc_id, text):
    embedding = get_ollama_embedding(text)
    if embedding is not None:
        collection.add(
            documents=[text],
            embeddings=[embedding],
            ids=[doc_id]
        )

def query_similar_documents(query, n_results=5):
    query_embedding = get_ollama_embedding(query)
    if query_embedding is not None:
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        return results
    return None


if __name__ == "__main__":
    # Example usage
    add_document("1", "Blockchain technology is associated with the financial industry, but it can be applied to other industries. The supporting architecture of blockchain has the immense potential to transform the delivery of healthcare, medical, clinical, and life sciences, due to the extended functionality and distinct features of its distributed ledger")
    add_document("2", "The potential scale of impact is comparable to that seen with the introduction of TCP/IP. Blockchain technology has captured the interest of healthcare providers and biomedical scientists within various healthcare domains such as longitudinal healthcare records, automated claims, drug development, interoperability in population health, consumer health, patient portals, medical research, data security, and reducing costs with supply chain management.")
    add_document("3", "It is not yet clear if blockchain is going to disrupt healthcare, but healthcare organizations are monitoring its potential closely for prospective concepts like secure patient IDs. Realistically, the adoption and implementation of blockchains will be a gradual evolution over time, but now is the time to take a fresh look at its possibilities in healthcare and biomedical sciences.")

    query = "What is blockchain can disrupt?"
    results = query_similar_documents(query)
    print("Query Results:", results)