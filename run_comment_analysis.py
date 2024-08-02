#!/usr/bin/env python
# coding: utf-8

# My own modules
from models.text_models import TextModelManager
from models.llm_api import LLM
from api.youtube_api import YoutubeAPI
from analysis.classification_analysis import ClassificationAnalyzer
from analysis.statements_analysis import StatementsAnalyzer
from analysis.clustering import ClusteringAnalyzer


# Logging
import logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',  # Define the log format with milliseconds
    datefmt='%Y-%m-%d %H:%M:%S'  # Define the date and time format without milliseconds
)
logger = logging.getLogger(__name__)


class AnalysisRunner:
    def __init__(self) -> None:
        # Initialize classification models
        self.text_model_manager = TextModelManager()

        # Set up LLM
        self.llm = LLM()

        # Youtube API
        self.youtube = YoutubeAPI()
        yt_video_id = self.pick_yt_video_id()
        self.youtube_setup(yt_video_id)

        # Get comments
        self.comments = self.youtube.get_comments(yt_video_id)

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
        # Classification Analysis
        classification_analyzer = ClassificationAnalyzer(self.comments)
        logger.info(classification_analyzer.run_all_analyses())

        # Clustering
        clustering_analyzer = ClusteringAnalyzer(video_id=self.youtube.get_current_video(), comments=self.comments)
        clustering_analyzer.cluster()
        clustering_analyzer.describe_clusters()

        # LLM Statement Extraction
        statements_analyzer = StatementsAnalyzer(
            video_id=self.youtube.get_current_video(),
            comments=self.comments
        )

        statements_analyzer.run_analysis(
            limit_statements=2,  # For testing, limit number of statements
            comment_top_k=2  # reduced count for testing
        )


if __name__ == "__name__":
    runner = AnalysisRunner()
    runner.run_all_analyses()
