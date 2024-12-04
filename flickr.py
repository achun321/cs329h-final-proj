import os
from datasets import load_dataset
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import faiss
from tqdm import tqdm
from PIL import Image

# Load the Flickr30k dataset
ds = load_dataset("nlphuji/flickr30k", split="test")

# Use a subset of the dataset (e.g., first 1000 examples)
subset_size = 1000
ds = ds.select(range(subset_size))

# Check if FAISS index exists
faiss_index_path = "flickr30k_image_index.faiss"
if os.path.exists(faiss_index_path):
    # Load FAISS index directly
    index = faiss.read_index(faiss_index_path)
    print(f"Loaded FAISS index with {index.ntotal} vectors.")
else:
    # Compute embeddings and create FAISS index
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    batch_size = 32
    image_embeddings = []
    images = []

    for i, example in enumerate(tqdm(ds)):
        images.append(example["image"])  # Collect images for embedding

        if len(images) == batch_size:
            # Compute image embeddings for the batch
            inputs = processor(images=images, return_tensors="pt").to(device)
            with torch.no_grad():
                embeds = model.get_image_features(**inputs)
                embeds = embeds / embeds.norm(dim=-1, keepdim=True)
                image_embeddings.append(embeds.cpu().numpy())
            images = []

    # Process remaining images
    if images:
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            embeds = model.get_image_features(**inputs)
            embeds = embeds / embeds.norm(dim=-1, keepdim=True)
            image_embeddings.append(embeds.cpu().numpy())

    # Concatenate embeddings and create FAISS index
    image_embeddings = np.concatenate(image_embeddings, axis=0).astype("float32")
    index = faiss.IndexFlatIP(image_embeddings.shape[1])  # Inner Product for cosine similarity
    index.add(image_embeddings)
    faiss.write_index(index, faiss_index_path)
    print(f"Created and saved FAISS index with {index.ntotal} vectors.")

# Perform a query with text embeddings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

query = "A group of people playing soccer"
inputs = processor(text=query, return_tensors="pt").to(device)
with torch.no_grad():
    query_embedding = model.get_text_features(**inputs)
    query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
query_embedding = query_embedding.cpu().numpy().astype("float32")

k = 5
distances, indices = index.search(query_embedding, k)

for i, idx in enumerate(indices[0]):
    idx = int(idx)  
    example = ds[idx]  
    image = example["image"]  
    caption = example["caption"]  

    retrieved_image_path = f"retrieved_images/retrieved_{i+1}.jpg"
    image.save(retrieved_image_path, "JPEG")  
    print(f"Rank {i+1}: {caption[0]} (Saved to {retrieved_image_path})")


