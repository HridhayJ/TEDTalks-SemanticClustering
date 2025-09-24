import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

# Load transcripts
transcripts = pd.read_csv("data/transcripts.csv")
talking = transcripts.iloc[0:50]["transcript"]
talking_list = talking.tolist()

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to split long transcripts into chunks
def chunk_text(text, chunk_size=500):
    curr_index = chunk_size
    last_index = 0
    chunks = []
    while len(text) > curr_index:
        while text[curr_index] != ' ' and curr_index > last_index:
            curr_index -= 1
        chunks.append(text[last_index:curr_index])
        last_index = curr_index
        curr_index += chunk_size
    chunks.append(text[last_index:])
    return chunks

# Compute embeddings
embeddings = []
for talk in talking_list:
    chunks = chunk_text(talk, 500)
    chunk_embedding = [model.encode(chunk) for chunk in chunks]
    talk_embedding = np.mean(chunk_embedding, axis=0)
    embeddings.append(talk_embedding)

embeddings = np.array(embeddings)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=0)
kmeans.fit(embeddings)
labels = kmeans.labels_

# Reduce dimensionality for visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(embeddings)

# Plot PCA scatter
plt.figure(figsize=(10, 7))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap='viridis')
plt.colorbar()
plt.title("PCA of TED Talk Embeddings")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.show()

# Print cluster samples
num_clusters = len(set(labels))
for cluster_id in range(num_clusters):
    print(f"\n=== Cluster {cluster_id} ===")
    cluster_indices = np.where(labels == cluster_id)[0]
    sample_indices = cluster_indices[:3]  # show first 3 talks in cluster
    for idx in sample_indices:
        print(f"--- Talk {idx} ---")
        print(talking_list[idx][:500] + "...")