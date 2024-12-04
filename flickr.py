import os
from datasets import load_dataset
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import faiss
from sklearn.preprocessing import normalize
from tqdm import tqdm
from PIL import Image

# Load the Flickr30k dataset
ds = load_dataset("nlphuji/flickr30k", split="test")

# Use a subset of the dataset (e.g., first 1000 examples)
subset_size = 1000
ds = ds.select(range(subset_size))

# Load CLIP model and processor, and move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Parameters
batch_size = 32  # Adjust based on available GPU memory
image_embeddings = []
text_embeddings = []

# Directories for saving results
image_folder = "flickr30k_images"
retrieved_images_folder = "retrieved_images"
os.makedirs(image_folder, exist_ok=True)
os.makedirs(retrieved_images_folder, exist_ok=True)

# Batch processing
images = []
captions = []

for i, example in enumerate(tqdm(ds)):
    # Save image to disk
    image = example["image"]
    image_path = os.path.join(image_folder, f"{i}.jpg")
    image.save(image_path, "JPEG")  # Save image as JPEG

    # Collect images and first captions for batching
    images.append(image)
    captions.append(example["caption"][0])  # First caption in the list

    # Process the batch if it reaches the batch_size
    if len(images) == batch_size:
        # Process image embeddings
        image_inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            image_embeds = model.get_image_features(**image_inputs)  # Image embeddings
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)  # Normalize
            image_embeddings.append(image_embeds.cpu().numpy())  # Move to CPU

        # Process text embeddings
        text_inputs = processor(text=captions, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            text_embeds = model.get_text_features(**text_inputs)  # Text embeddings
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)  # Normalize
            text_embeddings.append(text_embeds.cpu().numpy())  # Move to CPU

        # Clear batches
        images = []
        captions = []

# Process any remaining images and captions
if images:
    # Process remaining image embeddings
    image_inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        image_embeds = model.get_image_features(**image_inputs)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        image_embeddings.append(image_embeds.cpu().numpy())

    # Process remaining text embeddings
    text_inputs = processor(text=captions, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_embeds = model.get_text_features(**text_inputs)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        text_embeddings.append(text_embeds.cpu().numpy())

# Concatenate all batches into single arrays
image_embeddings = np.concatenate(image_embeddings, axis=0)
text_embeddings = np.concatenate(text_embeddings, axis=0)

print(f"Image Embeddings Shape: {image_embeddings.shape}")
print(f"Text Embeddings Shape: {text_embeddings.shape}")

# Normalize embeddings for FAISS
normalized_image_embeddings = normalize(image_embeddings, axis=1).astype("float32")

# Create FAISS index
dimension = normalized_image_embeddings.shape[1]  # Dimensionality of embeddings (1024 for ViT-L/14)
index = faiss.IndexFlatIP(dimension)  # Inner Product (cosine similarity with normalized embeddings)
index.add(normalized_image_embeddings)  # Add image embeddings to FAISS index
print(f"FAISS index contains {index.ntotal} vectors.")

# Perform a query with text embeddings
query = "A group of people playing soccer"  # Example query
query_inputs = processor(text=query, return_tensors="pt", padding=True).to(device)
with torch.no_grad():
    query_embedding = model.get_text_features(**query_inputs)
    query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
query_embedding = query_embedding.cpu().numpy().astype("float32")  # Convert to NumPy for FAISS

# Search the FAISS index
k = 5  # Number of nearest neighbors to retrieve
distances, indices = index.search(query_embedding, k)

# Save retrieved images to a folder
for i, idx in enumerate(indices[0]):
    # Retrieve dataset example
    idx = int(idx)
    image_path = os.path.join(image_folder, f"{idx}.jpg")  # Get saved image
    retrieved_image_path = os.path.join(retrieved_images_folder, f"retrieved_{i+1}.jpg")  # Save retrieved image
    caption = ds[idx]["caption"]  # Retrieve captions for the image

    # Copy image to retrieved_images folder
    with Image.open(image_path) as img:
        img.save(retrieved_image_path)

    # Print the rank, caption, and path
    print(f"Rank {i+1}: {caption[0]}")
    print(f"Saved to: {retrieved_image_path}")
