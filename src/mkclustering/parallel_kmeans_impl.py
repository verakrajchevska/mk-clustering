import numpy as np
from multiprocessing import Pool

class ParallelKMeans:
    def __init__(self, n_clusters, max_iter=300, num_cores=None):
        self.n_clusters = int(n_clusters)
        self.max_iter = int(max_iter)
        self.num_cores = int(num_cores) if num_cores else 1
        self.centroids = None
        self.iterations = 0

    def initialize_centroids(self, data: np.ndarray) -> np.ndarray:
        idx = np.random.choice(data.shape[0], size=self.n_clusters, replace=False)
        return data[idx]

    def euclidean_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.sqrt(np.sum((a - b) ** 2))

    def closest_centroid(self, point: np.ndarray) -> int:
        diffs = self.centroids - point  
        d2 = np.einsum("ij,ij->i", diffs, diffs)
        return int(np.argmin(d2))

    def assign_clusters(self, data_chunk: np.ndarray):
        clusters = [[] for _ in range(self.n_clusters)]
        for i in range(data_chunk.shape[0]):
            c = self.closest_centroid(data_chunk[i])
            clusters[c].append(data_chunk[i])
        return clusters

    def merge_clusters(self, clusters_by_chunk):
        merged = [[] for _ in range(self.n_clusters)]
        for chunk in clusters_by_chunk:
            for c_idx, cl in enumerate(chunk):
                if cl:
                    merged[c_idx].extend(cl)
        return merged

    def compute_centroids(self, clusters, data: np.ndarray) -> np.ndarray:
        new = []
        for cl in clusters:
            if len(cl) > 0:
                new.append(np.mean(np.stack(cl, axis=0), axis=0))
            else:
                new.append(data[np.random.randint(data.shape[0])])
        C = np.vstack(new).astype(data.dtype)
        C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
        return C

    def fit(self, data: np.ndarray):
        self.centroids = self.initialize_centroids(data)
        for it in range(self.max_iter):
            data_perm = np.random.permutation(data)
            chunks = np.array_split(data_perm, self.num_cores)

            if self.num_cores > 1:
                with Pool(self.num_cores) as pool:
                    clusters_by_chunk = pool.map(self.assign_clusters, chunks)
            else:
                clusters_by_chunk = [self.assign_clusters(ch) for ch in chunks]

            clusters = self.merge_clusters(clusters_by_chunk)
            new_centroids = self.compute_centroids(clusters, data)

            if np.allclose(self.centroids, new_centroids, atol=1e-6):
                self.centroids = new_centroids
                self.iterations = it + 1
                return self
            self.centroids = new_centroids

        self.iterations = self.max_iter
        return self

    def predict(self, data: np.ndarray):
        labels = np.empty(data.shape[0], dtype=np.int32)
        for i in range(data.shape[0]):
            labels[i] = self.closest_centroid(data[i])
        return labels
