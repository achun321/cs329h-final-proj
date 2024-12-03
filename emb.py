from datasets import load_dataset
import faiss
import numpy as np
from sklearn.preprocessing import normalize


ds = load_dataset("nousr/laion5b-subset-and-cliph-embeddings")
embeddings = np.load("laion5b-subset-and-cliph-embeddings/embeddings/img_emb_0.npy")

# normalize embeddings + create FAISS index
normalized_embeddings = normalize(embeddings, axis=1).astype('float32')
dimension = normalized_embeddings.shape[1]  
index = faiss.IndexFlatIP(dimension)  

# Add embeddings to the index
index.add(normalized_embeddings)
print(f"Index contains {index.ntotal} vectors.")
