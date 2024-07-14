from datetime import datetime
from typing import List
from models.computations import ClassificationType
from models.text_models import TextModelManager


class Comment:
    def __init__(self, author: str, text: str, time: datetime, likes: int, replies: List):
        # Basic info
        self.author = author
        self.text = text
        self.time = time
        self.likes = likes
        self.replies = replies

        # For this comment to calculate its own classification results and embedding, it needs a TextModelManager
        self._text_model_manager = TextModelManager()

        # Classifier output (about this comment)
        self._classifier_res = {}

        # Text embedding
        self._embedding = None

    def get_classification(self, classi_type: ClassificationType):
        if classi_type not in self._classifier_res:
            self._classifier_res[classi_type] = self._text_model_manager.classify(self.text, classi_type)
        return self._classifier_res[classi_type]
    
    def get_embedding(self):
        if self._embedding is None:
            self._embedding = self._text_model_manager.embed(self.text)
        return self._embedding

    def __repr__(self):
        return f"Comment({self.author} @ {self.time.isoformat()}: '{self.text}') ({self.likes} likes; {len(self.replies)} replies)"

    def __str__(self):
        return self.__repr__()