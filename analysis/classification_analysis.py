from typing import List, Dict
import numpy as np
from tqdm import tqdm


from structures.comment import Comment, flatten_comments
from models.computations import ClassificationType


class ClassificationAnalyzer:
    def __init__(self, comments: List[Comment]) -> None:
        self._comments_toplevel = comments
        self._comments_flattened = flatten_comments(comments)
    
    def _find_mean_classes(self, comments: List[Comment], classi_type: ClassificationType, take_argmax=False):
        res = []
        likes = []
        for comm in tqdm(
            comments,
            desc=f"Determining computated facts ({classi_type.name}; argmax={take_argmax}) ..."
        ):
            comp = dict(comm.get_classification(classi_type))

            # If requested, set the maximum-probability class to the sum of probabilities and the rest to 0.0
            if take_argmax:
                max_idx = np.argmax(comp.values())
                prob_sum = sum(comp.values())
                
                for idx, k in enumerate(comp.keys()):
                    if idx == max_idx:
                        comp[k] = prob_sum
                    else:
                        comp[k] = 0.0
                
            res.append(comp)
            likes.append(comm.likes)

        # Find out like-weighted score for each class
        classes = res[0].keys()
        stats = {}
        for cl in classes:
            # Get raw scores
            class_scores = np.array([scores[cl] for scores in res])

            # Weight each score by likes
            class_scores = np.multiply(class_scores, np.array(likes))

            # Find out mean score
            total_likes = np.sum(likes)
            mean_score = np.sum(class_scores) / total_likes

            # Save result
            stats[cl] = mean_score

        return stats

    def mean_classification_analysis(self, comments: List[Comment], classi_type: ClassificationType) -> Dict:
        res = {}

        for argmax_label, argmax_bool in [("soft", False), ("hard", True)]:
            res[argmax_label] = self._find_mean_classes(comments, classi_type, take_argmax=argmax_bool)
        
        return res
    
    def show_extreme_class_examples(self, classi_type: ClassificationType, num_shown : int = 10) -> Dict:
        res = {'num_shown': num_shown}

        r = res['res'] = {}
        for cl in self._comments_flattened[0].get_classification(classi_type).keys():
            r[cl] = [
                str(comm) for comm in
                sorted(
                    self._comments_flattened,
                    key=lambda comm: comm.get_classification(classi_type)[cl], reverse=True
                )[:num_shown]
            ]

        return res
    
    @staticmethod
    def _classes_to_arr(comm: Comment, classi_type: ClassificationType):
        return np.array(list(comm.get_classification(classi_type).values()))
    
    def class_disagreement_in_replies(self, classi_type: ClassificationType, num_rows_output: int = 100) -> List:
        # Gather data
        rows = []
        for comm in self._comments_toplevel:
            # Skip if there are no replies
            if len(comm.replies) == 0:
                continue
        
            # Consider the first reply
            repl = comm.replies[0]
        
            # Find out sum of differences between sentiment of comment and reply
            diffs = np.sum(np.power(np.abs(self._classes_to_arr(comm, classi_type) - self._classes_to_arr(repl, classi_type)), 2))
            
            rows.append({"comment": comm.text, "reply": repl.text, "difference": diffs})
        
        # Sort rows by difference - highest first
        rows.sort(key=lambda entry: entry["difference"], reverse=True)

        # Limit output rows
        if num_rows_output is not None:
            rows = rows[:num_rows_output]

        return rows
    
    def classification_analysis(self, classi_type: ClassificationType) -> Dict:
        res = {}

        # Mean classes
        r = res['mean'] = {}
        for comment_label, comment_list in [("top-level", self._comments_toplevel), ("all", self._comments_flattened)]:
            r[comment_label] = self.mean_classification_analysis(comment_list, classi_type)
        
        # Extreme examples
        res['extreme'] = self.show_extreme_class_examples(classi_type)

        # Disagreement in replies
        res['disagreement'] = self.class_disagreement_in_replies(classi_type)

        return res
    
    def run_all_analyses(self) -> Dict:
        res = {}

        res['info'] = "All results are weighted by comment likes."

        r = res['res'] = {}
        for classi_type in ClassificationType:
            r[classi_type.name] = self.classification_analysis(classi_type)

        return res