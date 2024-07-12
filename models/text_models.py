from transformers import pipeline
from sentence_transformers import SentenceTransformer
from detoxify import Detoxify
import torch
import numpy as np


from .computations import ClassificationType, classification_length_limits
from util.string_utils import split_text_if_long


class TextModelManager:
    def __init__(self):
        # Cache instanced classification models and functions
        self._models = {}
        self._classi_funs = {}

        # Set up torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # Embedding model
        self._embedding_model = None
        self._emb_cache = {}

    def _get_function(self, classi_type: ClassificationType):
        # Load model if necessary
        if classi_type not in self._models:

            if classi_type == ClassificationType.Sentiment:
                # Model
                model = pipeline(
                    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", 
                    top_k=None
                )
                # Function
                func = lambda text: {cl["label"]: cl["score"] for cl in model(text)[0]}

            elif classi_type == ClassificationType.Toxicity:
                # Model
                model = Detoxify('original', device=self._device)
                # Function
                func = lambda text: model.predict(text)

            elif classi_type == ClassificationType.Emotion:
                # Model
                model = pipeline(
                    task="text-classification",
                    model="SamLowe/roberta-base-go_emotions",
                    top_k=None
                )
                # Function
                func = lambda text: {cl["label"]: cl["score"] for cl in model(text)[0]}

            # Save model and function
            self._models[classi_type] = model
            self._classi_funs[classi_type] = func

        # Return function
        return self._classi_funs[classi_type]

    def classify(self, text: str, classi_type: ClassificationType):
        # Text may be very long. Split the comment into parts. For most comments, the result will be a single part
        len_limit = classification_length_limits[classi_type]
        text_parts = split_text_if_long(text, max_len=len_limit)

        # Calculate function for each part
        fun = self._get_function(classi_type)
        res = []
        for part in text_parts:
            res.append(fun(part))

        # Aggregate the results
        if len(res) > 1:
            if type(res[0]) == dict:
                res = {k: np.mean([r[k] for r in res]) for k in res[0].keys()}
            else:
                res = np.mean(res, axis=0)
        else:
            res = res[0]
        
        return res
    
    def embed(self, text, use_cache=True):
        # Load embedding from cache if possible
        if use_cache and text in self._emb_cache:
            return self._emb_cache[text]

        # Load model if necessary
        if self._embedding_model is None:
            # Models
            # multilingual model: SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            #  - this is the first model I tried
            #  - It worked reasonably well but I saw that it mapped phrases and their negated form very closely to each other
            #
            # multilingual model: SentenceTransformer("sentence-transformers/LaBSE")
            #  - second model I tried.
            #  - works about as well as the first
            #
            # monolingual (English only) model: SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self._embedding_model = SentenceTransformer("sentence-transformers/LaBSE")
        
        # Embed text
        emb = self._embedding_model.encode(text)
        emb /= np.linalg.norm(emb)  # normalize to unit length

        # Save into cache
        if use_cache:
            self._emb_cache[text] = emb

        return emb