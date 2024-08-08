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

    def _create_summary(self):
        g = 14
        # TODO: Save information in a more structured manner into report - e.g., it's really hard to find the % of positive and negative comments
        # TODO: Collect important information from the raw results
        # TODO: Put all of it into a prompt
        # TODO: Create a summary using the LLM
        pass

    def summarize(self):
        # Create summary
        summary = self._create_summary()

        # Save summary
        self._report.summary = summary

    

