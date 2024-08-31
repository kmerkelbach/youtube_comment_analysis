#!/usr/bin/env python
# coding: utf-8

import argparse
import os

import numpy as np
import pandas as pd

from api.youtube_api import YoutubeAPI
# My own modules
from structures.report import Report
from execution.run_comment_analysis import AnalysisRunner

# Logging
import logging

logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
    # Define the log format with milliseconds
    datefmt='%Y-%m-%d %H:%M:%S'  # Define the date and time format without milliseconds
)
logger = logging.getLogger(__name__)


# CSV fields
field_url = "URL"
field_title = "Title"
field_creator = "Creator"
field_num_comments = "# Comments"
field_summary = "Summary"
field_llm_total_chars = "# LLM Characters"
field_video_id = "Video ID"



class CSVRunner:
    def __init__(self, csv_path: str) -> None:
        # Open CSV file
        self._input_path = csv_path
        self._state = pd.read_csv(self._input_path)
        logger.info(f"Opened CSV file from {self._input_path} with {len(self._state)} rows.")

        # Figure out the video ID
        self._state[field_video_id] = self._state[field_url].apply(YoutubeAPI.video_id_for_youtube_url)

        # Figure out which videos still need to be done
        self.videos_remaining = []
        for _, row in self._state.iterrows():
            video_id = row[field_video_id]

            # Test if the title field is filled
            if not (field_title in row and not np.isnan(row[field_title]) and len(row[field_title]) > 0):
                self.videos_remaining.append(video_id)

        logger.info(f"Of {len(self._state)} entries in the input file, {len(self.videos_remaining)} are remaining.")

    def start(self):
        for video_id in self.videos_remaining:
            runner = self._run_for_video_id(video_id)
            self._add_results_to_state(runner)

        logger.info("All video analyses done.")

        # Once all videos are done, write new CSV to disk
        self._state.to_csv(
            self._input_path,
            index=False
        )

    def _add_results_to_state(self, runner: AnalysisRunner):
        # Find row of state table corresponding to this video
        video_id = runner.yt_video_id
        row_index = self._state[self._state[field_video_id] == video_id].index[0]

        # Extract some results
        title = runner.youtube.get_title()
        creator = runner.youtube.get_creator_name()
        num_comments = runner.get_report().get_comment_count()
        llm_call_stats = runner.get_report().get_llm_stats()
        llm_total_sum = int(sum(llm_call_stats.values()))
        summary_text = runner.get_report().summary

        # Write results to the table
        for field, content in [(field_title, title), (field_creator, creator), (field_num_comments, num_comments),
                               (field_summary, summary_text), (field_llm_total_chars, llm_total_sum)]:
            # Add column if it isn't present in state
            if field not in self._state:
                self._state[field] = content.__class__()

            # Add entry at target row
            self._state.loc[row_index, field] = content

    def _run_for_video_id(self, video_id: str) -> AnalysisRunner:
        runner = AnalysisRunner(yt_video_id=video_id)
        runner.run_all_analyses()
        return runner

def parse_args():
    parser = argparse.ArgumentParser("YouTube Comment Analyzer - CSV Mode")
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help=f"Path to CSV file with at least the column `{field_url}` containing YouTube video URLs"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    runner = CSVRunner(csv_path=args.csv_path)
    runner.start()
