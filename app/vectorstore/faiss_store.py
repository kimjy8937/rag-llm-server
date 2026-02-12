import faiss
import numpy as np
import os
import pickle

class FaissStore:
    def __init__(self, dimension, index_path="index/faiss.index", meta_path="index/docs.pkl"):
        self.index_path = index_path
        self.meta_path = meta_path

        if os.path.exists(index_path) and os.path.exists(meta_path):
            print("기존 FAISS 인덱스 로드")
            self.index = faiss.read_index(index_path)

            with open(meta_path, "rb") as f:
                self.documents = pickle.load(f)
        else:
            print("새 FAISS 인덱스 생성")
            self.index = faiss.IndexFlatL2(dimension)
            self.documents = []

    def add(self, embeddings, documents):
        self.index.add(np.array(embeddings))
        self.documents.extend(documents)

        faiss.write_index(self.index, self.index_path)

        with open(self.meta_path, "wb") as f:
            pickle.dump(self.documents, f)

        print("FAISS 인덱스 저장 완료")

    def search(self, query_embedding, k=2):
        distances, indices = self.index.search(query_embedding, k)
        return [self.documents[i] for i in indices[0]]
