from typing import List, Dict, Optional
from tqdm import tqdm
import numpy as np
import itertools
from sklearn.cluster import KMeans, SpectralClustering, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples
import matplotlib as mpl
import matplotlib.pyplot as plt
import umap.umap_ as umap
from sklearn.manifold import TSNE

from structures.comment import Comment, flatten_comments, sample_from_comments
from util.string_utils import post_process_single_entry_json
from api.youtube_api import YoutubeAPI
from models.llm_api import LLM


import logging
logger = logging.getLogger(__name__)


class Clustering:
    def __init__(self, labels: np.ndarray, labels_unique: List[int], silhouette_by_label: Dict[int, float], clustering_function=None, topics: Dict[int, str] = None):
        self.labels = labels
        self.labels_unique = labels_unique
        self.num_clusters = len(self.labels_unique)
        self.silhouette_by_label = silhouette_by_label
        self.silhouette_coef = np.mean(list(silhouette_by_label.values()))
        self.clustering_function = clustering_function
        self.topics = topics


class ClusteringAnalyzer:
    def __init__(self, video_id: str, comments: List[Comment]) -> None:
        self.video_id = video_id

        self._youtube = YoutubeAPI()
        self._video_title = self._youtube.get_title(self.video_id)

        self._llm = LLM()

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

        # Find topics of each cluster
        self._find_cluster_topics(clustering)

        return clustering
    
    def plot_clustering(self, clustering: Clustering, use_umap=True):
        # Prepare colormap for plotting
        cm_steps = len(clustering.labels_unique)
        hsv = mpl.colormaps.get_cmap('hsv')
        cmap = mpl.colors.ListedColormap(hsv(np.linspace(0,1,cm_steps + 1)[:-1]))
        
        # Reduce the dimensionality of the clustered points (embedding vectors)
        if use_umap:
            # UMAP
            reducer = umap.UMAP()
        else:
            # t-SNE
            reducer = TSNE(
                n_components=2,
                learning_rate='auto',
                init='random',
                perplexity=3
            )

        # Fit
        matrix_2d = reducer.fit_transform(self._get_emb_matrix())

        # Plot
        colors = [cmap.colors[clustering.labels_unique.index(lab)] for lab in clustering.labels]
        plt.scatter(x=matrix_2d[:, 0], y=matrix_2d[:, 1], c=colors)

    def _build_prompt_find_topic(self, comments: List[Comment]):
        lines = [f"You are a professional YouTube comment analyst. Given a video title and some comments," \
                 " find the topic of the comments."]
        lines.append(f"Video title: {self._video_title}")
        
        lines.append("\nSample from the comments:")
        comm_lines = sample_from_comments(comments)
        lines += comm_lines

        lines.append("\nExtract a single, coherent topic that these comments are discussing. The topic you find can also" \
                     " be about the style or mood of the comments. " \
                    "A topic should be a simple notion, e.g., \"Jokes\" or \"Choosing a keyboard\"." \
                    "There is no need to repeat the video title in your assessment. The topic should also describe what the" \
                    " comments are saying, so it shouldn't be, e.g., \"Reactions to Video\" or anything generic of that sort." \
                    " Provide your assessment in the form of JSON such as {\"topic\": your_topic_goes_here}.")

        prompt = "\n".join(lines)
        return prompt

    def _find_cluster_topics(self, clustering: Clustering):
        matrix = self._get_emb_matrix()
        topics = {}
        for label in tqdm(clustering.labels_unique, desc="Find cluster topics ..."):
            # Get indices
            clus_indices = np.where(clustering.labels == label)[0]

            # Find mean embedding of cluster
            clus_mean_emb = np.mean(np.stack([matrix[idx] for idx in clus_indices]), axis=0)

            # Sort comments by their distance to the mean embedding of the cluster
            clus_comments = [self._comments[idx] for idx in clus_indices]
            clus_comments.sort(key=lambda comment: np.sum(np.abs(comment.get_embedding()) - clus_mean_emb))

            # Determine a central comment topic using the LLM
            clus_comments_central = clus_comments[:1000]
            prompt = self._build_prompt_find_topic(clus_comments_central)
            res_raw = self._llm.chat(prompt)
            topics[int(label)] = post_process_single_entry_json(res_raw)
        clustering.topics = topics


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