from typing import List, Dict, Optional, Tuple
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
from models.math_funcs import cos_sim
from api.youtube_api import YoutubeAPI
from models.text_models import TextModelManager
from models.llm_api import LLM


import logging
logger = logging.getLogger(__name__)


class Clustering:
    def __init__(self, labels: np.ndarray, labels_unique: List[int], silhouette_by_label: Dict[int, float], clustering_function=None, topics: Dict[int, str] = None):
        self.labels = labels
        self.labels_unique = [int(l) for l in labels_unique]
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

        self._text_model_manager = TextModelManager()

        self._comments = flatten_comments(comments)
        self._emb_matrix = None

        self._clustering = None

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
        self._clustering = self._pick_clustering(clusterings)

        # Find topics of each cluster
        self._find_cluster_topics()

        # Fuse clusters by topic
        self._fuse_clusters_by_topic()

    def describe_clusters(self, show_random_comments: Optional[int] = 50):
        res = {}

        for lab in self._clustering.labels_unique:
            r = res[lab] = {}

            # Topic
            r['topic'] = self._clustering.topics[lab]

            # Size
            labels = self._clustering.labels
            cluster_size = int(sum(labels == lab))
            r['size'] = {
                'abs': cluster_size,
                'rel': cluster_size / len(labels)
            }

            # Get indices
            clus_indices = np.where(labels == lab)[0]

            # Show random comments
            if show_random_comments is not None and show_random_comments > 0:
                r['random_comments'] = []

                rnd_indices = np.random.choice(clus_indices, size=min(show_random_comments, cluster_size), replace=False)
                for idx in rnd_indices:
                    comment = self._comments[idx]
                    r['random_comments'].append(comment)
        
        return res

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

    def _fuse_clusters_embedding_sim(self, clustering_topics: Dict[int, str]) -> List[Tuple[int, str]]:
        cluster_groups = [[]]

        # Fuse clusters into groups based on embedding similarity of the topic
        embed = self._text_model_manager.embed
        for lab, topic in clustering_topics.items():
            # Store this cluster label and topic as a tuple
            tup = (lab, topic)
            
            # Try to find a spot for this topic in one of the groups
            found_group = False
            for group in cluster_groups:

                # If the group is empty, add the cluster (this only happens at the beginning)
                if len(group) == 0:
                    group.append(tup)
                    found_group = True
                    break

                # Compare this cluster's embedding with the group
                mean_sim = np.mean([cos_sim(embed(top), embed(topic)) for (l, top) in group])
                if mean_sim > 0.55:
                    group.append(tup)
                    found_group = True
                    break

            # If we already found a group, go on to the next cluster's topic
            if found_group:
                continue

            # Start a new group
            cluster_groups.append([tup])
        
        return cluster_groups
    
    def _synthesize_fused_topic_names(self, cluster_groups: List[Tuple[int, str]]) -> List[Tuple[List[int], str]]:
        fused_groups = []
        for group in tqdm(cluster_groups, desc="Fusing groups ..."):
            labs, topics = zip(*group)

            if len(topics) > 1:
                prompt = self._build_prompt_fuse_topics(topics)
                res_raw = self._llm.chat(prompt)
                topic = post_process_single_entry_json(res_raw)
            else:
                topic = topics[0]

            fused_groups.append((labs, topic))
        
        return fused_groups
    
    def _reassign_grouped_clusters(self, fused_groups: List[Tuple[List[int], str]]) -> None:
        clustering = self._clustering
        clustering.topics = {}  # reset topics dictionary - we will fill it with the new topics
        for label_group, topic in fused_groups:
            # No need to change any labels if the "group" doesn't have multiple labels
            if len(label_group) <= 1:
                clustering.topics[label_group[0]] = topic
                continue

            # Paint all labels in group to match the first label
            label_group = list(label_group)
            lab_first = label_group.pop(0)
            for lab in label_group:
                clustering.labels[np.where(clustering.labels == lab)] = lab_first

            # Save new topic in dictionary
            clustering.topics[lab_first] = topic

            # Update unique labels
            clustering.labels_unique = [int(l) for l in list(np.unique(clustering.labels))]

        def rename_in_dict(d, pre, post):
            d[post] = d[pre]
            return d

        # Assign cluster labels such that there are no gaps
        labels_no_gaps = list(np.arange(len(clustering.labels_unique)))
        if clustering.labels_unique != labels_no_gaps:
            for label_pre, label_post in zip(clustering.labels_unique, labels_no_gaps):
                clustering.labels[np.where(clustering.labels == label_pre)] = label_post

                clustering.topics = rename_in_dict(clustering.topics, label_pre, label_post)
                clustering.silhouette_by_label = rename_in_dict(clustering.silhouette_by_label, label_pre, label_post)

            clustering.labels_unique = labels_no_gaps
            clustering.num_clusters = len(clustering.labels_unique)

    def _fuse_clusters_by_topic(self) -> None:
        # Fuse based on embedding distance/similarity of topics
        cluster_groups = self._fuse_clusters_embedding_sim(self._clustering.topics)
        
        # Give fused groups new names
        fused_groups = self._synthesize_fused_topic_names(cluster_groups)
        
        # Change labeling of clustering to reflect group fusions
        self._reassign_grouped_clusters(fused_groups)

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
    
    def _build_prompt_fuse_topics(self, topics: List[str]):
        lines = [f"You are a professional YouTube comment analyst. Given a video title and some comment topics," \
                 " find a new description of the topic that reflects the core concept of the listed topics."]
        lines.append(f"Video title: {self._video_title}")
        
        lines.append("\nComment topics:")
        lines += [f"- {t}" for t in topics]

        lines.append("\nExtract a single, coherent topic that describes all these topics. The topic you find can also" \
                     " be about the style or mood of the comments. " \
                    "A topic should be a simple notion, e.g., \"Jokes\" or \"Choosing a keyboard\"." \
                    "There is no need to repeat the video title in your assessment. The topic shouldn't be," \
                        " e.g., \"Reactions to Video\" or anything generic of that sort. Provide your assessment" \
                            " in the form of JSON such as {\"topic\": your_topic_goes_here}.")

        prompt = "\n".join(lines)
        return prompt

    def _find_cluster_topics(self):
        matrix = self._get_emb_matrix()
        topics = {}
        for label in tqdm(self._clustering.labels_unique, desc="Find cluster topics ..."):
            # Get indices
            clus_indices = np.where(self._clustering.labels == label)[0]

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
        self._clustering.topics = topics


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