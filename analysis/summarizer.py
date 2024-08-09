from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import itertools


from api.youtube_api import YoutubeAPI
from models.llm_api import LLM
from structures.report import Report
from structures.comment import Comment
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



        # TODO: Gather important information I want the LLM to know about
        # TODO: Experiment with prompting styles - the summary should be captivating and not feel LLM-written
        g = 15

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

    

