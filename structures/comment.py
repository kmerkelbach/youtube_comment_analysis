from datetime import datetime
from typing import List
import numpy as np

from util.string_utils import truncate_line
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

    def num_replies(self):
        return len(self.replies)

    def __repr__(self):
        return f"Comment({self.author} @ {self.time.isoformat()}: '{self.text}') ({self.likes} likes; {len(self.replies)} replies)"

    def __str__(self):
        return self.__repr__()


def flatten_comments(comments: List[Comment]):
    res = []
    for comm in comments:
        # Add comment itself
        res.append(comm)

        # Add its replies
        res += flatten_comments(comm.replies)
    
    return res


def sample_from_comments(comments: List[Comment], max_chars_per_comment: int = 200, max_comment_chars_shown: int = 2500) -> List[str]:
    # Create a shuffled list of comment indices
    indices = np.arange(len(comments))
    np.random.shuffle(indices)

    # Sample comments until we have enough characters
    double_newline = "\n\n"
    comm_lines = []
    for idx in indices:

        # Break if we have too many characters
        if sum(len(l) for l in comm_lines) >= max_comment_chars_shown:
            break

        # Get comment text
        text = comments[idx].text

        # Remove extraneous newlines
        while double_newline in text:
            text = text.replace(double_newline, "\n")

        # Truncate text if necessary
        if len(text) > max_chars_per_comment:
            text = truncate_line(text, max_chars_per_comment)

        # Add the comment text
        comm_lines.append(f"- \"{text}\"")
    
    return comm_lines