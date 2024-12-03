from datasets import load_dataset
import faiss
import numpy as np
from sklearn.preprocessing import normalize
from transformers import CLIPProcessor, CLIPModel
import torch


ds = load_dataset("nousr/laion5b-subset-and-cliph-embeddings")
embeddings = np.load("laion5b-subset-and-cliph-embeddings/embeddings/img_emb_0.npy")

# normalize embeddings + create FAISS index
normalized_embeddings = normalize(embeddings, axis=1).astype('float32')
dimension = normalized_embeddings.shape[1]  
index = faiss.IndexFlatIP(dimension)  

# Add embeddings to the index
index.add(normalized_embeddings)
print(f"Index contains {index.ntotal} vectors.")


# Load CLIP model for text queries
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_text_embedding(query):
    inputs = processor(text=query, return_tensors="pt")
    with torch.no_grad():
        text_embedding = model.get_text_features(**inputs)
    return normalize(text_embedding.numpy())

# query LAION
query = "Community"
query_embedding = get_text_embedding(query).reshape(1, -1)
distances, indices = index.search(query_embedding, k=5)

# mapping back to dataset
for idx in indices[0]:
    print(f"Result: {ds[idx]['__key__']}, URL: {ds[idx]['__url__']}")