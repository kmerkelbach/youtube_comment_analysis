#!/usr/bin/env python
# coding: utf-8

import argparse

# Logging
import logging

from analysis.classification_analysis import ClassificationAnalyzer
from analysis.clustering import ClusteringAnalyzer
from analysis.statements_analysis import StatementsAnalyzer
from analysis.summarizer import ReportSummarizer
from api.youtube_api import YoutubeAPI
from models.llm_api import LLM

# Internal modules
from models.text_models import TextModelManager
from structures.comment import flatten_comments
from structures.report import Report

logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
    # Define the log format with milliseconds
    datefmt='%Y-%m-%d %H:%M:%S'  # Define the date and time format without milliseconds
)
logger = logging.getLogger(__name__)


class AnalysisRunner:
    def __init__(self, yt_video_id: str) -> None:
        # Initialize classification models
        self.text_model_manager = TextModelManager()

        # Set up LLM
        self.llm = LLM()

        # Youtube API
        self.youtube = YoutubeAPI()
        self.yt_video_id = yt_video_id
        self.youtube_setup(self.yt_video_id)

        # Get comments
        self.comments = self.youtube.get_comments()

        # Start empty report
        self._report = Report(
            video_id=self.yt_video_id
        )

    def get_report(self):
        return self._report

    def youtube_setup(self, yt_video_id: str):
        self.youtube.set_current_video(yt_video_id)

        # Print some information about the video
        logger.info(self.youtube.get_title())
        logger.info(self.youtube.get_creator_name())

    @staticmethod
    def pick_yt_video_id():
        yt_video_test_id_tomato = "9WQnap-UAiQ"
        yt_video_test_id_10k_comments = "2-XxbdR3Nik"
        yt_video_test_id_4500_comments = "-ih0B9yn32Q"
        yt_video_test_id_4k_comments_beard_meets_schnitzel = "qPd9qPUR2_U"
        yt_video_test_id_2000_comments = "rX2tK-qSVpk"
        yt_video_test_id_700_comments = "VCXqELB3UPg"
        yt_video_test_id_300_comments = "yQqJafC7xv0"
        yt_video_test_id_25_comments = "kiF0wgM8zGc"
        yt_video_test_id_50_comments = "LHQMIuzjl48"

        yt_video_id = yt_video_test_id_25_comments
        return yt_video_id

    def run_all_analyses(self):
        # General facts
        self._report._comment_count_toplevel = len(self.comments)
        self._report._comment_count = len(flatten_comments(self.comments))

        # Clustering
        clustering_analyzer = ClusteringAnalyzer(video_id=self.yt_video_id, comments=self.comments)
        clustering_analyzer.cluster()
        clus_res = clustering_analyzer.describe_clusters()
        self._report.res_clustering = clus_res

        # Classification Analysis
        classification_analyzer = ClassificationAnalyzer(self.comments)
        class_res = classification_analyzer.run_all_analyses()
        self._report.res_classification = class_res

        # LLM Statement Extraction
        statements_analyzer = StatementsAnalyzer(
            video_id=self.yt_video_id,
            comments=self.comments
        )

        statement_res = statements_analyzer.run_analysis(
            comment_top_k=50  # 50 -> around 0.35 USD LLM cost; between 500 and 750 LLM comparisons
        )
        self._report.res_statements = statement_res

        # Summarize report
        summarizer = ReportSummarizer(
            video_id=self.yt_video_id,
            comments=self.comments,
            report=self._report
        )
        summarizer.summarize()

        # Save report
        self._report.save_to_disk()


def parse_args():
    parser = argparse.ArgumentParser("YouTube Comment Analyzer")
    parser.add_argument(
        "--video_id",
        type=str,
        required=True,
        help="YouTube video ID of the video to be analyzed"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    runner = AnalysisRunner(yt_video_id=args.video_id)
    runner.run_all_analyses()
