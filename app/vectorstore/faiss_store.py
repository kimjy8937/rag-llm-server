import faiss
import numpy as np

class FaissStore:
    def __init__(self, dimension):
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []

    def add(self, embeddings, documents):
        self.index.add(np.array(embeddings))
        self.documents.extend(documents)

    def search(self, query_embedding, k=2):
        distances, indices = self.index.search(query_embedding, k)
        return [self.documents[i] for i in indices[0]]
