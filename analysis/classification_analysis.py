from typing import List
import numpy as np
import pandas as pd
from tqdm import tqdm


from structures.comment import Comment
from models.computations import ClassificationType


class ClassificationAnalyzer:
    def __init__(self, comments: List[Comment]) -> None:
        self._comments_toplevel = comments
        self._comments_flattened = self._flatten_comments(comments)

    def _flatten_comments(self, comments: List[Comment]):
        res = []
        for comm in comments:
            # Add comment itself
            res.append(comm)

            # Add its replies
            res += self._flatten_comments(comm.replies)
        
        return res
    
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

    def mean_classification_analysis(self, comments: List[Comment], classi_type: ClassificationType) -> str:
        out_lines = []

        for argmax_label, argmax_bool in [("Soft", False), ("Hard", True)]:
            mean_clss = self._find_mean_classes(comments, classi_type, take_argmax=argmax_bool)
        
            out_lines.append(f"({argmax_label}) Mean {classi_type.name} for {len(comments)} comments:")
            for cl, mc in mean_clss.items():
                out_lines.append(f"{cl}:".ljust(20) + f"{100 * mc:0.2f}%".rjust(6))
        
        return "\n".join(out_lines)
    
    def show_extreme_class_examples(self, classi_type: ClassificationType, num_shown : int = 10) -> str:
        out_lines = []

        for cl in self._comments_flattened[0].get_classification(classi_type).keys():
            out_lines.append(f"{num_shown} most {cl} comments: ")
            out_lines += [
                str(comm) for comm in
                sorted(
                    self._comments_flattened,
                    key=lambda comm: comm.get_classification(classi_type)[cl], reverse=True
                )[:num_shown]
            ]
            out_lines.append("")

        return "\n".join(out_lines)
    
    @staticmethod
    def _classes_to_arr(comm: Comment, classi_type: ClassificationType):
        return np.array(list(comm.get_classification(classi_type).values()))
    
    def class_disagreement_in_replies(self, classi_type: ClassificationType) -> pd.DataFrame:
        df_rows = []
        for comm in self._comments_toplevel:
            # Skip if there are no replies
            if len(comm.replies) == 0:
                continue
        
            # Consider the first reply
            repl = comm.replies[0]
        
            # Find out sum of differences between sentiment of comment and reply
            diffs = np.sum(np.power(np.abs(self._classes_to_arr(comm, classi_type) - self._classes_to_arr(repl, classi_type)), 2))
            
            df_rows.append({"comment": comm.text, "reply": repl.text, "difference": diffs})
        
        df = pd.DataFrame(df_rows)
        df = df.sort_values(by="difference", ascending=False)
        return df
    
    def classification_analysis(self, classi_type: ClassificationType) -> str:
        out_lines = []

        # Mean classes
        for comment_label, comment_list in [("top-level", self._comments_toplevel), ("all", self._comments_flattened)]:
            out_lines.append(f"Classification ({classi_type.name}) analysis for {comment_label} comments:")
            out_lines.append(self.mean_classification_analysis(comment_list, classi_type))
            out_lines.append("")
        
        # Extreme examples
        out_lines.append(self.show_extreme_class_examples(classi_type))

        # Disagreement in replies
        out_lines.append(f"Disagreement in replies for {classi_type.name} classes:")
        df = self.class_disagreement_in_replies(classi_type).iloc[:15]
        out_lines.append(df.to_markdown())

        out_str = "\n".join(out_lines)
        return out_str