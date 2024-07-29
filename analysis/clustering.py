from typing import List, Dict, Optional
from tqdm import tqdm
import numpy as np
import itertools
from sklearn.cluster import KMeans, SpectralClustering, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples


from structures.comment import Comment, flatten_comments


import logging
logger = logging.getLogger(__name__)


class Clustering:
    def __init__(self, labels: np.ndarray, labels_unique: List[int], silhouette_by_label: Dict[int, float], clustering_function):
        self.labels = labels
        self.labels_unique = labels_unique
        self.num_clusters = len(self.labels_unique)
        self.silhouette_by_label = silhouette_by_label
        self.silhouette_coef = np.mean(list(silhouette_by_label.values()))
        self.clustering_function = clustering_function


class ClusteringAnalyzer:
    def __init__(self, comments: List[Comment]) -> None:
        self._comments = flatten_comments(comments)
        self._emb_matrix = None

    def _get_emb_matrix(self):
        if self._emb_matrix is None:
            emb_vecs = []
            for comm in tqdm(self._comments, desc="Calculating embeddings ..."):
                emb_vecs.append(comm.get_embedding())
            self._emb_matrix = np.stack(emb_vecs)
        return self._emb_matrix
    
    @staticmethod
    def _eval_clustering(matrix, labels):
        labs_unique = list(np.unique(labels))
        
        # Silhouette score for each sample (i.e., comment)
        try:
            sil_all = silhouette_samples(matrix, labels)
        except ValueError:
            # this may happen if there is only one label
            sil_all = np.copy(labels)
            sil_all.fill(-1)  # worst possible value
        
        # Silhouette score, aggregated by cluster
        sil_by_label = {lab: np.mean(sil_all[np.where(labels == lab)[0]]) for lab in labs_unique}
        
        return labs_unique, sil_by_label
    
    def _generate_clusterings(self) -> List[Clustering]:
        # Clustering settings
        n_range = [2, 3, 4, 5, 6, 7, 8, 16, 32, 64]
        n_range = [n for n in n_range if n < len(self._comments)]  # remove cluster counts larger than the number of samples
        clus_funs = [cluster_kmeans, cluster_gmm, cluster_spectral_clustering, cluster_hdbscan]

        clusterings = []
        matrix = self._get_emb_matrix()
        for n, clus_fun in tqdm(list(itertools.product(n_range, clus_funs)), desc="Clustering ..."):
            # Cluster
            labels = clus_fun(matrix, n=n)

            # Evaluate
            labs_unique, sil_by_label = self._eval_clustering(matrix, labels)

            # Skip clustering if it is degenerate (i.e., the majority of points are in a single cluster)
            cluster_fractions = [len(np.where(labels == lab)[0]) / len(labels) for lab in labs_unique]
            num_clusters = len(labs_unique)
            frac_limit = min(2 / num_clusters, 0.8)
            if max(cluster_fractions) > frac_limit:
                continue
            
            # Add clustering
            clustering = Clustering(
                labels=labels,
                labels_unique=labs_unique,
                silhouette_by_label=sil_by_label,
                clustering_function=clus_fun
            )
            clusterings.append(clustering)
        
        return clusterings
    
    def _pick_clustering(self, clusterings: List[Clustering]) -> Clustering:
        # Sort by mean of Silhouette coefficient: largest first
        clusterings.sort(key=lambda clus: clus.silhouette_coef, reverse=True)

        clus_best = clusterings[0]
        logger.info(f"Best clustering out of {len(clusterings)} is with {clus_best.num_clusters} clusters, with "
                    f"a mean Silhouette coefficient of {clus_best.silhouette_coef} (function was {clus_best.clustering_function}).")
        return clus_best

    def cluster(self):
        # Get candidate clusterings
        clusterings = self._generate_clusterings()

        # Pick a clustering
        clustering = self._pick_clustering(clusterings)

        return clustering


def cluster_kmeans(matrix, n=5):
    clustering_method = KMeans(n_clusters=n)
    clustering_method.fit(matrix)
    return clustering_method.labels_


def cluster_spectral_clustering(matrix, n=5):
    clustering_method = SpectralClustering(n_clusters=n)
    clustering_method.fit(matrix)
    return clustering_method.labels_


def cluster_hdbscan(matrix, n=5):
    # argument `n` is ignored
    clustering_method = HDBSCAN()
    clustering_method.fit(matrix)
    return clustering_method.labels_


def cluster_gmm(matrix, n=5):
    clustering_method = GaussianMixture(n_components=n)
    clustering_method.fit(matrix)
    labels = clustering_method.predict(matrix)
    return labels