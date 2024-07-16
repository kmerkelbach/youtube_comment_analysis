from typing import List, Dict
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict


from structures.comment import Comment
from models.computations import ClassificationType
from models.math_funcs import max_class
from util.string_utils import truncate_line, post_process_extract_statements, post_process_single_entry_json
from util.file_utils import named_dir
from api.youtube_api import YoutubeAPI
from models.llm_api import LLM


DOUBLE_NEWLINE = "\n\n"


class StatementsAnalyzer:
    def __init__(self, video_id: str, comments: List[Comment]) -> None:
        self.video_id = video_id
        self._comments = comments
        self._youtube = YoutubeAPI()
        self._llm = LLM()

        # Settings for prompt of agreement assessment
        self.agreement_prompt_settings = {
            "min": -5,
            "neut": 0,
            "max": 5
        }

        # Store results of statement agreement assessment - I can later use these samples to train a cheaper model
        self._data_dir = named_dir("agreements")

        # Grouped comments
        self._sentiment_groups = None
    
    def _group_comments(self):
        sentiment_groups = defaultdict(list)

        for comm in tqdm(self._comments, "Grouping by sentiment ..."):
            comp = comm.get_classification(ClassificationType.Sentiment)
            sen = max_class(comp)
            magnitude = comp[sen]
            if magnitude > 0.25:
                sentiment_groups[sen].append(comm)
        
        # Save groups
        self._sentiment_groups = sentiment_groups
    
    def _sample_from_comments(self, comments: List[Comment], max_chars_per_comment: int = 200, max_comment_chars_shown: int = 2500) -> List[str]:
        # Create a shuffled list of comment indices
        indices = np.arange(len(comments))
        np.random.shuffle(indices)

        # Sample comments until we have enough characters
        comm_lines = []
        for idx in indices:

            # Break if we have too many characters
            if sum(len(l) for l in comm_lines) >= max_comment_chars_shown:
                break

            # Get comment text
            text = comments[idx].text

            # Remove extraneous newlines
            while DOUBLE_NEWLINE in text:
                text = text.replace(DOUBLE_NEWLINE, "\n")

            # Truncate text if necessary
            if len(text) > max_chars_per_comment:
                text = truncate_line(text, max_chars_per_comment)

            # Add the comment text
            comm_lines.append(f"- \"{text}\"")
        
        return comm_lines

    def _build_prompt_extract_statements(self, comments: List[Comment]) -> str:
        title = self._youtube.get_title(self.video_id)
        lines = [f"You are a professional YouTube comment analyst. Given a video title and some comments, extract statements from the comments."]
        lines.append(f"Video title: {title}")
        
        lines.append("\nSample from the comments:")
        comm_lines = self._sample_from_comments(comments)
        lines += comm_lines

        lines.append("\nExtract 5 statements voiced in the comments. A statement should be a simple thought expressed by many comments, e.g., \"The video was well-edited.\" or \"I disagree with the premise of the video.\". Phrase each statement in a way it could be uttered by a viewer of the video. " \
                    "Do not explain any of the statements you extract. " \
                    "There is no need to repeat the video title in your assessment.")

        prompt = "\n".join(lines)
        return prompt
    
    def build_prompt_do_statements_agree(self, statement_1: str, statement_2: str) -> str:
        title = self._youtube.get_title(self.video_id)
        lines = ["You are a professional YouTube video comment analyst. Given a video title and a statement (or a comment) about the video, decide if the statements two agree. Note that it may be possible for a statement or a comment to express the desire for change or to voice disagreement."]
        lines.append(f"Video title: {title}")
        lines.append(f"Statement 1: {statement_1}")
        lines.append(f"Statement 2: {statement_2}")

        lines.append("\nA statement is a simple thought expressed by many comments, e.g., \"The video was well-edited.\" or \"I disagree with the premise of the video.\".")
        lines.append(f"First, think step by step about the two statements to determine if they agree. Finally, give your assessment of the agreement on a scale of {self.agreement_prompt_settings['min']} for total disagreement to {self.agreement_prompt_settings['max']} for total agreement. " \
                    f"The number {self.agreement_prompt_settings['neut']} is for unrelated statements (those which discuss different matters). Even if the sentiments of the statements are opposite: If they discuss different matters, the assessment should be {self.agreement_prompt_settings['neut']}. " \
                    "Provide your assessment in the form of JSON such as {\"agreement\": your_number_goes_here}.")

        prompt = "\n".join(lines)
        return prompt

    def extract_statements(self, min_comments: int = 10):
        # Group comments by sentiment
        self._group_comments()

        # Extract statements from positive and negative comments
        comment_statements = {}
        for sen, sen_comments in self._sentiment_groups.items():
            # Continue if we don't have enough comments
            if len(sen_comments) < min_comments:
                continue

            # Construct the LLM prompt and summarize the comments
            prompt = self._build_prompt_extract_statements(sen_comments)
            res_raw = self._llm.chat(prompt)
            res_lines = post_process_extract_statements(res_raw)
            comment_statements[sen] = res_lines
        
        return comment_statements