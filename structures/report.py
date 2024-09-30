import os
from datetime import datetime

from models.llm_api import LLM
from util.file_utils import save_json, named_dir


class Report:
    def __init__(self, video_id: str) -> None:
        # Video information
        self._video_id = video_id

        # Statistics
        self._time_started = datetime.now().isoformat()
        self._time_finished = None
        self._llm = LLM()  # need the LLM for getting its input/output statistics

        # General
        self._comment_count = None
        self._comment_count_toplevel = None

        # Results
        self.res_classification = None
        self.res_clustering = None
        self.res_statements = None

        # Summary
        self.summary = None

    def _construct_file_path(self):
        return os.path.join(
            named_dir("reports"),
            f"{self._video_id}_text_data.json"
        )

    def _convert_random_comments_to_str(self, rnd_field="random_comments"):
        res_clus = self.res_clustering
        for clus_label in res_clus.keys():
            clus_info = res_clus[clus_label]
            if rnd_field in clus_info:
                comments = clus_info[rnd_field]
                comments = [str(comm) for comm in comments]
                clus_info[rnd_field] = comments

    def get_comment_count(self):
        return self._comment_count

    def get_llm_stats(self):
        return {
            "num_input_chars": self._llm.get_num_input_chars(),
            "num_output_chars": self._llm.get_num_output_chars()
        }

    def save_to_disk(self):
        # Add finishing time
        self._time_finished = datetime.now().isoformat()

        # Convert random comments in clustering into strings
        self._convert_random_comments_to_str()

        # Construct JSON-seriazable dictionary
        report_info = {
            "meta": {
                "time_started": self._time_started,
                "time_finished": self._time_finished,
                "num_comments": self._comment_count
            },
            "llm_use": self.get_llm_stats(),
            "results": {
                "classification": self.res_classification,
                "clustering": self.res_clustering,
                "statements": self.res_statements
            },
            "summary": self.summary
        }

        try:
            save_json(
                file_path=self._construct_file_path(),
                data=report_info
            )
        except:
            from IPython import embed
            embed()
