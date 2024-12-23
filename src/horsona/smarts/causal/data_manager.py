import json
from typing import Generic, TypeVar

import numpy as np

from horsona.index.ollama_model import OllamaEmbeddingModel

T = TypeVar("T")

eps = 1e-10


class DataManager(Generic[T]):
    def __init__(self):
        self.data_types = {}
        self._data: list[dict[str, T]] = []
        self.embedding_model = OllamaEmbeddingModel("imcurie/bge-large-en-v1.5")

    def get_representative_points(
        self, columns: set[str], max_points: int, required: list[set[str]] = []
    ) -> list[dict[str, T]]:
        data = []
        for datapoint in self._data:
            restricted_data = {k: v for k, v in datapoint.items() if k in columns}

            matching_requirements = []
            for requirement in required:
                if set(restricted_data.keys()).intersection(requirement):
                    matching_requirements.append(requirement)
            if len(matching_requirements) != len(required):
                continue

            data.append(restricted_data)

        if len(data) <= max_points:
            return data

        embeddings = self.embedding_model.get_data_embeddings(
            [json.dumps(d) for d in data]
        )
        selections = kmeans(embeddings, max_points, distance_fn=cosine_distances)
        representative_points = [data[i] for i in selections]

        return representative_points

    def extend(self, data: list[dict[str, T]]):
        self._data.extend(data)


def euclidean_distances(X, centers):
    """Compute Euclidean distances between points and centers."""
    return ((X[:, np.newaxis, :] - centers) ** 2).sum(axis=2)


def cosine_distances(X, centers):
    """Compute cosine distances between points and centers."""
    # Normalize the input vectors
    X_normalized = X / np.linalg.norm(X, axis=1)[:, np.newaxis]
    centers_normalized = centers / np.linalg.norm(centers, axis=1)[:, np.newaxis]

    # Compute cosine similarity
    similarities = np.dot(X_normalized, centers_normalized.T)

    # Clip values to [-1, 1] to handle numerical errors
    similarities = np.clip(similarities, -1.0, 1.0)

    # Convert to angles in radians using arccos
    return np.arccos(similarities)


def _init_centroids(X, rng, n_clusters, distance_fn):
    """Initialize cluster centers using k-means++ algorithm."""
    n_samples, n_features = X.shape
    centers = np.zeros((n_clusters, n_features))

    # Choose first center randomly
    centers[0] = X[rng.randint(n_samples)]

    # Compute distances to the initial center
    distances = distance_fn(X, centers[:1])

    # Choose remaining centers
    for k in range(1, n_clusters):
        # Choose next center proportional to distance squared
        probs = distances / (distances.sum() + eps)
        cumprobs = np.cumsum(probs)

        found = False
        while not found:
            r = rng.rand()

            for j, p in enumerate(cumprobs):
                if r < p:
                    centers[k] = X[j]
                    found = True
                    break

        if k < n_clusters - 1:
            # Update distances for remaining iterations
            new_distances = distance_fn(X, centers[k : k + 1])
            distances = np.minimum(distances, new_distances)

    return centers


def _single_kmeans(X, rng, n_clusters, max_iter, distance_fn):
    """Run single k-means clustering."""
    centers = _init_centroids(X, rng, n_clusters, distance_fn)

    for iteration in range(max_iter):
        # Assign points to nearest centroid
        distances = distance_fn(X, centers)
        labels = np.argmin(distances, axis=1)

        # Update centroids
        new_centers = np.array([X[labels == k].mean(axis=0) for k in range(n_clusters)])

        # Handle empty clusters
        empty_clusters = np.where(np.isnan(new_centers).any(axis=1))[0]
        if len(empty_clusters) > 0:
            furthest_points = np.argmax(distances, axis=1)
            for cluster_idx in empty_clusters:
                new_point_idx = furthest_points[cluster_idx]
                new_centers[cluster_idx] = X[new_point_idx]

        # Check for convergence
        centers = new_centers

    inertia = distances[np.arange(len(X)), labels].sum()
    return labels, inertia


def kmeans(X, n_clusters, distance_fn, n_init=10, max_iter=300):
    """
    Perform k-means clustering with custom distance metric.

    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
        Training instances to cluster.
    n_clusters : int
        Number of clusters to form.
    metric : str, default='euclidean'
        The distance metric to use. Options are:
        - 'euclidean': Standard Euclidean distance
        - 'cosine': Cosine distance (1 - cosine similarity)
    n_init : int, default=10
        Number of times the k-means algorithm will be run with different
        centroid seeds.
    max_iter : int, default=300
        Maximum number of iterations for each run.

    Returns:
    --------
    labels : array of shape (n_samples,)
        Index of the cluster each sample belongs to.
    """
    X = np.asarray(X)
    rng = np.random.RandomState()

    best_inertia = None
    best_labels = None

    for i in range(n_init):
        labels, inertia = _single_kmeans(X, rng, n_clusters, max_iter, distance_fn)

        if best_inertia is None or inertia < best_inertia:
            best_labels = labels
            best_inertia = inertia

    return best_labels
