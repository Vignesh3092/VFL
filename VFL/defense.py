import numpy as np
from sklearn.cluster import KMeans
import torch

def apply_kmeans_defense(embeddings):
    # embeddings: list of torch.Tensor (batch_size, embedding_dim)
    vectors = np.stack([e.mean(axis=0).detach().cpu().numpy() for e in embeddings])
    kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto').fit(vectors)
    labels_kmeans = kmeans.labels_
    majority_label = np.argmax(np.bincount(labels_kmeans))

    # Keep only clients in the majority cluster
    selected_embeddings = [
        emb for emb, label in zip(embeddings, labels_kmeans) if label == majority_label
    ]
    return selected_embeddings if selected_embeddings else embeddings  # fallback if empty
