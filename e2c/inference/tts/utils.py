from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer #外部库
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
_EMBEDDING_MODELS = {}

def get_embedding_model(model_name: str) -> SentenceTransformer:

    if model_name not in _EMBEDDING_MODELS:
        try:
            _EMBEDDING_MODELS[model_name] = SentenceTransformer(model_name)
        except Exception as e:
            raise IOError(
                f"无法加载嵌入模型 '{model_name}'. "
                "请确保install sentence-transformers"
                f"并且模型名称正确。错误: {e}"
            )
    return _EMBEDDING_MODELS[model_name]


def embed_plans(
        plans: List[str],
        model_name: str = "all-mpnet-base-v2",
) -> np.ndarray:

    if not plans:
        return np.array([])

    model = get_embedding_model(model_name)
    embeddings = model.encode(plans, convert_to_numpy=True)
    return embeddings


def cluster_plans(
        embeddings: np.ndarray,
        num_clusters: int,
) -> Tuple[List[int], List[int]]:

    num_samples = embeddings.shape[0]
    if num_samples == 0:
        return [], []

    if num_samples <= num_clusters:
        centroid_indices = list(range(num_samples))
        cluster_sizes = [1] * num_samples
        return centroid_indices, cluster_sizes

    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(embeddings)

    centroid_indices = []
    cluster_sizes = []

    for i in range(num_clusters):
        cluster_member_indices = np.where(labels == i)[0]
        if len(cluster_member_indices) == 0:
            continue
        cluster_sizes.append(len(cluster_member_indices))
        centroid_vector = kmeans.cluster_centers_[i]
        member_embeddings = embeddings[cluster_member_indices]
        distances = euclidean_distances(member_embeddings, centroid_vector.reshape(1, -1))
        closest_member_local_idx = np.argmin(distances)
        closest_member_global_idx = cluster_member_indices[closest_member_local_idx]
        centroid_indices.append(closest_member_global_idx)

    return centroid_indices, cluster_sizes