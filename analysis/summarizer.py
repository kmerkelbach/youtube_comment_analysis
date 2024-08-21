from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import itertools


from api.youtube_api import YoutubeAPI
from models.llm_api import LLM
from structures.report import Report
from structures.comment import Comment, sample_from_comments
from util.string_utils import format_large_number


import logging
logger = logging.getLogger(__name__)


class ReportSummarizer:
    def __init__(self, video_id: str, comments: List[Comment], report: Report) -> None:
        # Set up APIs
        self.video_id = video_id
        self._comments = comments

        self._youtube = YoutubeAPI()
        self._video_title = self._youtube.get_title(self.video_id)

        self._llm = LLM()

        # Remember report
        self._report = report

    def _prompt_summary(self):
        lines = ["You are a professional YouTube video comment analyst. Given a video title and analytical statements made about the comments on the video, summarize the analysis."]
        lines.append("")

        # General facts
        lines.append("General:")
        lines.append(f"Video title: {self._video_title}")
        lines.append(f"Comment count: {format_large_number(self._report._comment_count)}")
        lines.append("")

        # Sentiment analysis
        lines.append("Sentiment analysis:")
        sentiments = self._report.res_classification['res']['Sentiment']['mean']['all']['hard']
        # e.g., {'neutral': 0.1, 'positive': 0.8, 'negative': 0.1}
        for sen_name, sen_frac in sentiments.items():
            lines.append(f"{100 * sen_frac:0.1f}% of comments are {sen_name}")
        lines.append("")

        # Statement extraction
        lines.append("Statements extracted from the comments:")
        s_voices = self._report.res_statements['voices']
        s_scores = self._report.res_statements['scores']
        score_ranges = self._report.res_statements['rules']['prompt_ranges']
        for idx, (statement, voice_info) in enumerate(s_voices.items()):
            score = s_scores[statement]
            s_lines = []
            s_lines.append(f"Statement {idx + 1}: {statement}")
            s_lines.append(f"Agreement score is {score:0.0f} (a value between {score_ranges['min']} and {score_ranges['max']}).")
            s_lines.append(f"{100 * voice_info['frac_engaged']:0.0f}% of comments are talking about this.")
            if 'opinions' in voice_info:
                s_lines.append(f"Out of those, " +
                               ", ".join([f"{100 * fraction:0.0f}% {opinion}"
                                          for (opinion, fraction) in voice_info['opinions'].items()]))
            s_lines.append("")
            lines += s_lines

        # Clustering by topic
        lines.append("Clusters of topics:")
        clustering = self._report.res_clustering
        lines.append(f"Comments were clustered by topic and we identified {len(clustering)} clusters in the comments.")
        for clus_label, clus_info in clustering.items():
            c_lines = []
            c_lines.append(f"Cluster {clus_label + 1}: {clus_info['topic']}")
            c_lines.append(f"{100 * clus_info['size']['rel']:0.0f}% of comments are in this cluster.")
            c_lines.append("Here are some random comments from this cluster:")
            c_lines += sample_from_comments(
                clus_info['random_comments'],
                max_comment_chars_shown=600
            )
            c_lines.append("")
            lines += c_lines

        # TODO: Instruct the LLM to write a compelling summary with all of the important facts.
        # TODO: Experiment with prompting styles - the summary should be captivating and not feel LLM-written
        g = 15

    def _build_prompt_extract_statements(self, comments: List[Comment]) -> str:
        lines = [
            f"You are a professional YouTube comment analyst. Given a video title and some comments, extract statements from the comments."]
        lines.append(f"Video title: {self._video_title}")

        lines.append("\nSample from the comments:")
        comm_lines = sample_from_comments(comments)
        lines += comm_lines

        lines.append(
            "\nExtract 5 statements voiced in the comments. A statement should be a simple thought expressed by many comments, e.g., \"The video was well-edited.\" or \"I disagree with the premise of the video.\". Phrase each statement in a way it could be uttered by a viewer of the video. " \
            "Do not explain any of the statements you extract. " \
            "There is no need to repeat the video title in your assessment.")

        prompt = "\n".join(lines)
        return prompt

    def _create_summary(self):
        # Make prompt and send to LLM
        prompt = self._prompt_summary()
        res = self._llm.chat(prompt)

        return res

    def summarize(self):
        # Create summary
        summary = self._create_summary()

        # Save summary
        self._report.summary = summary

    

