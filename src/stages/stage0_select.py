from __future__ import annotations
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans


def clip_cluster_select(
    image_paths: list[str],
    k: int,
    model_name: str,
    pretrained: str,
) -> list[str]:
    """Phase A: Select k diverse photos using CLIP embeddings + K-means."""
    import open_clip
    import torch

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model.eval()

    images = torch.stack([
        preprocess(Image.open(p).convert("RGB")) for p in image_paths
    ])

    with torch.no_grad():
        features = model.encode_image(images).cpu().numpy().astype(np.float32)

    features /= np.linalg.norm(features, axis=1, keepdims=True)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)

    selected: list[str] = []
    for cluster_id in range(k):
        mask = labels == cluster_id
        cluster_feats = features[mask]
        center = kmeans.cluster_centers_[cluster_id]
        dists = np.linalg.norm(cluster_feats - center, axis=1)
        local_best = int(np.argmin(dists))
        global_idx = int(np.where(mask)[0][local_best])
        selected.append(image_paths[global_idx])

    return selected
