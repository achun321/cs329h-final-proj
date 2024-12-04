import os
from datasets import load_dataset
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import faiss
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import networkx as nx
import json
import openai




def load_dataset_subset(subset_size=1000):
    """
    Load the Flickr30k dataset and select a subset of it.
    """
    ds = load_dataset("nlphuji/flickr30k", split="test")
    return ds.select(range(subset_size))


def compute_image_embeddings(ds, model, processor, batch_size=32):
    """
    Compute image embeddings for the given dataset.
    """
    image_embeddings = []
    images = []

    for i, example in enumerate(tqdm(ds)):
        images.append(example["image"])  # Collect images for embedding

        if len(images) == batch_size:
            # Process a batch of images
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

    return np.concatenate(image_embeddings, axis=0).astype("float32")


def create_or_load_faiss_index(embeddings, faiss_index_path):
    """
    Create a FAISS index with the given embeddings or load an existing index.
    """
    if os.path.exists(faiss_index_path):
        index = faiss.read_index(faiss_index_path)
        print(f"Loaded FAISS index with {index.ntotal} vectors.")
    else:
        index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner Product for cosine similarity
        index.add(embeddings)
        faiss.write_index(index, faiss_index_path)
        print(f"Created and saved FAISS index with {index.ntotal} vectors.")
    return index


def create_similarity_graph(indices, embeddings):
    """
    Create a graph for the retrieved nodes based on their cosine similarity.

    Args:
        indices: List of indices of the top-k retrieved nodes.
        embeddings: Corresponding embeddings of the retrieved nodes.
    
    Returns:
        G: A NetworkX graph where nodes are the retrieved indices and edges
           are weighted by their cosine similarity.
    """
    G = nx.Graph()

    # Add nodes with default weight
    for idx in indices:
        G.add_node(idx, weight=1.0)

    # Compute cosine similarity for edges
    num_nodes = len(indices)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):  # Avoid redundant pairs and self-loops
            similarity = np.dot(embeddings[i], embeddings[j])
            G.add_edge(indices[i], indices[j], weight=similarity)

    return G


def query_faiss_index(query, model, processor, index, ds, k=5):
    """
    Perform a query using the FAISS index and retrieve the top-k results.
    Also create a similarity graph for the retrieved results.
    """
    inputs = processor(text=query, return_tensors="pt").to(device)
    with torch.no_grad():
        query_embedding = model.get_text_features(**inputs)
        query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
    query_embedding = query_embedding.cpu().numpy().astype("float32")

    # Search the FAISS index
    distances, indices = index.search(query_embedding, k)

    # Retrieve embeddings for the top-k indices
    top_k_embeddings = []
    for idx in indices[0]:
        top_k_embeddings.append(index.reconstruct(int(idx)))
    top_k_embeddings = np.array(top_k_embeddings)

    # Create a similarity graph
    G = create_similarity_graph(indices[0], top_k_embeddings)
    print(f"Created a graph with {len(G.nodes)} nodes and {len(G.edges)} edges.")

    # Retrieve and display results
    for i, idx in enumerate(indices[0]):
        idx = int(idx)
        example = ds[idx]
        image = example["image"]
        caption = example["caption"]

        # Save the retrieved image
        retrieved_image_path = f"retrieved_images/retrieved_{idx}.jpg"
        os.makedirs("retrieved_images", exist_ok=True)
        image.save(retrieved_image_path, "JPEG")

        print(f"Rank {i+1}: {caption[0]} (Saved to {retrieved_image_path})")

    return G  # Return the graph

def save_graph_image(G, save_path="retrieved_images/retrieved_graph.png", title="Graph Visualization"):
    """
    Save the graph as an image to the specified path.
    
    Args:
        G: NetworkX graph to visualize.
        save_path: Path to save the graph image.
        title: Title of the graph visualization.
    """
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure directory exists

    # Draw the graph
    pos = nx.spring_layout(G, seed=42)  # Spring layout for better aesthetics
    plt.figure(figsize=(10, 8))
    
    # Draw nodes and edges with weights
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=700,
        node_color="skyblue",
        edge_color="gray",
        font_weight="bold"
    )
    
    # Add edge weights
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    # Title and save
    plt.title(title, fontsize=15)
    plt.savefig(save_path, format="png")
    plt.close()  # Close the figure to avoid display-related issues
    print(f"Graph saved to {save_path}")
    
def normalize_graph_weights(G, node_range=(0, 1), edge_range=(0, 1)):
    """
    Normalize node and edge weights to lie within specified ranges.

    Args:
        G: NetworkX graph.
        node_range: Tuple specifying min and max for node weights.
        edge_range: Tuple specifying min and max for edge weights.
    """
    # Normalize node weights
    node_min, node_max = node_range
    for node in G.nodes:
        weight = G.nodes[node]["weight"]
        G.nodes[node]["weight"] = max(node_min, min(node_max, weight))

    # Normalize edge weights
    edge_min, edge_max = edge_range
    for u, v in G.edges:
        weight = G[u][v]["weight"]
        G[u][v]["weight"] = max(edge_min, min(edge_max, weight))
        
def find_top_j_closest_nodes(G, j=3):
    """
    Find the top-j closest nodes in the graph based on edge weights.
    
    Args:
        G: NetworkX graph.
        j: Number of nodes to retrieve.

    Returns:
        top_j_nodes: List of node indices for the top-j closest nodes.
    """
    if j > len(G.nodes):
        raise ValueError("j cannot be greater than the number of nodes in the graph.")

    # Compute closeness scores
    closeness_scores = []
    for node in G.nodes:
        total_weight = sum(G[node][neighbor]["weight"] for neighbor in G.neighbors(node))
        closeness_scores.append((node, total_weight))
    
    # Sort by closeness and select top-j
    closeness_scores.sort(key=lambda x: x[1], reverse=True)
    top_j_nodes = [node for node, _ in closeness_scores[:j]]
    return top_j_nodes

def generate_consolidated_caption(captions, api_key):
    """
    Use ChatGPT to consolidate multiple captions into a single caption.

    Args:
        captions: List of captions to consolidate.
        api_key: OpenAI API key.

    Returns:
        consolidated_caption: A single consolidated caption.
    """
    openai.api_key = api_key

    # Prepare prompt
    prompt = (
        "The following are captions of images related to a query. "
        "Please consolidate them into a single caption that captures their essence:\n\n"
    )
    prompt += "\n".join(f"- {caption}" for caption in captions)
    prompt += "\n\nConsolidated Caption:"

    # Call ChatGPT
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    consolidated_caption = response["choices"][0]["message"]["content"].strip()
    return consolidated_caption




def integrate_human_feedback(G, feedback_path, scaling_factor=2.0):
    """
    Integrate human feedback into the graph.

    Args:
        G: NetworkX graph.
        feedback_path: Path to the JSON file containing human feedback.
        scaling_factor: Factor to amplify the impact of feedback on node weights.
    """
    # Load feedback
    with open(feedback_path, "r") as f:
        feedback = json.load(f)

    # Update nodes based on feedback
    for row in feedback:
        img_num = row["img_num"]  # Node index
        score = row["score"]      # Relevance score
        
        if img_num in G.nodes:
            # Adjust node weight
            G.nodes[img_num]["weight"] += score * scaling_factor

            # Propagate feedback to neighbors
            for neighbor in G.neighbors(img_num):
                edge_weight = G[img_num][neighbor]["weight"]
                # Adjust edge weight based on feedback and similarity
                G[img_num][neighbor]["weight"] += score * edge_weight * scaling_factor
        else:
            print(f"Node {img_num} not found in graph. Skipping.")
    
    # Normalize node and edge weights
    normalize_graph_weights(G)

def calculate_average_cosine_similarity(G):
    """
    Calculate the average cosine similarity (edge weight) in the graph.

    Args:
        G: NetworkX graph.

    Returns:
        average_similarity: The average edge weight in the graph.
    """
    edge_weights = [G[u][v]["weight"] for u, v in G.edges]
    if not edge_weights:
        return 0.0  # Avoid division by zero if no edges
    return sum(edge_weights) / len(edge_weights)


if __name__ == "__main__":
    subset_size = 1000
    faiss_index_path = "flickr30k_image_index.faiss"
    feedback_path = "feedback.json"
    api_key = "your-openai-api-key"  # Replace with your OpenAI API key
    max_iterations = 1  # Number of requery cycles
    k = 10  # Number of top results to retrieve
    j = 3  # Number of top nodes to consolidate

    # Load dataset
    ds = load_dataset_subset(subset_size=subset_size)

    # Check if FAISS index exists, else compute embeddings and create it
    if not os.path.exists(faiss_index_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

        # Compute embeddings
        image_embeddings = compute_image_embeddings(ds, model, processor)
        index = create_or_load_faiss_index(image_embeddings, faiss_index_path)
    else:
        # Load FAISS index directly
        index = create_or_load_faiss_index(None, faiss_index_path)

    # Perform the initial query
    query = "Peace"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    # Initialize graph
    G = query_faiss_index(query, model, processor, index, ds, k=k)

    # Save the initial graph
    save_graph_image(G, save_path="retrieved_images/retrieved_graph_iteration_0.png")

    # Log average cosine similarity
    average_similarities = []
    avg_similarity = calculate_average_cosine_similarity(G)
    average_similarities.append(avg_similarity)
    print(f"Iteration 0: Average Cosine Similarity = {avg_similarity:.4f}")

    # Start the requery loop
    for iteration in range(1, max_iterations + 1):
        print(f"\n--- Iteration {iteration} ---\n")

        # Integrate human feedback into the graph
        integrate_human_feedback(G, feedback_path)

        # Save the graph after feedback
        save_graph_image(G, save_path=f"retrieved_images/retrieved_graph_iteration_{iteration}_post_feedback.png")

        # Calculate and log average cosine similarity
        avg_similarity = calculate_average_cosine_similarity(G)
        average_similarities.append(avg_similarity)
        print(f"Iteration {iteration}: Average Cosine Similarity = {avg_similarity:.4f}")

        # Find top-j closest nodes based on the updated graph
        top_j_nodes = find_top_j_closest_nodes(G, j=j)
        top_j_captions = [ds[int(node)]["caption"][0] for node in top_j_nodes]
        print(f"Top-{j} captions for iteration {iteration}: {top_j_captions}")
        raise Exception("TEST")

        # Generate a consolidated caption using ChatGPT
        new_caption = "A calm and so"
        print(f"Consolidated Caption for iteration {iteration}: {new_caption}")

        # Perform a re-query with the new caption
        G = query_faiss_index(new_caption, model, processor, index, ds, k=k)

        # Save the updated graph
        save_graph_image(G, save_path=f"retrieved_images/retrieved_graph_iteration_{iteration}_new_query.png")

    # Log overall average cosine similarities
    print("\n--- Average Cosine Similarities Over Iterations ---")
    for i, similarity in enumerate(average_similarities):
        print(f"Iteration {i}: {similarity:.4f}")






