from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import itertools


from structures.comment import Comment, sample_from_comments
from models.computations import ClassificationType
from models.math_funcs import max_class
from util.string_utils import post_process_extract_statements, post_process_single_entry_json
from util.file_utils import save_snippet
from api.youtube_api import YoutubeAPI
from models.llm_api import LLM


import logging
logger = logging.getLogger(__name__)


class StatementsAnalyzer:
    def __init__(self, video_id: str, comments: List[Comment], agreement_prompt_magnitude: int = 5) -> None:
        self.video_id = video_id
        self._comments = comments

        self._youtube = YoutubeAPI()
        self._video_title = self._youtube.get_title(self.video_id)

        self._llm = LLM()

        # Settings for prompt of agreement assessment
        self.agreement_prompt_settings = {
            "min": -agreement_prompt_magnitude,
            "neut": 0,
            "max": agreement_prompt_magnitude
        }
        self.voice_pos = "agree"
        self.voice_neut = "neutral"
        self.voice_neg = "disagree"
        self.voices_nonneut = (self.voice_pos, self.voice_neg)

        # Store results of statement agreement assessment - I can later use these samples to train a cheaper model
        self._data_dir_name = "agreements"

        # Cached results
        self._sentiment_groups = None
        self._comment_statements = None
    
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

    def _build_prompt_extract_statements(self, comments: List[Comment]) -> str:
        lines = [f"You are a professional YouTube comment analyst. Given a video title and some comments, extract statements from the comments."]
        lines.append(f"Video title: {self._video_title}")
        
        lines.append("\nSample from the comments:")
        comm_lines = sample_from_comments(comments)
        lines += comm_lines

        lines.append("\nExtract 5 statements voiced in the comments. A statement should be a simple thought expressed by many comments, e.g., \"The video was well-edited.\" or \"I disagree with the premise of the video.\". Phrase each statement in a way it could be uttered by a viewer of the video. " \
                    "Do not explain any of the statements you extract. " \
                    "There is no need to repeat the video title in your assessment.")

        prompt = "\n".join(lines)
        return prompt
    
    def _build_prompt_do_statements_agree(self, statement_1: str, statement_2: str) -> str:
        lines = ["You are a professional YouTube video comment analyst. Given a video title and a statement (or a comment) about the video, decide if the statements two agree. Note that it may be possible for a statement or a comment to express the desire for change or to voice disagreement."]
        lines.append(f"Video title: {self._video_title}")
        lines.append(f"Statement 1: {statement_1}")
        lines.append(f"Statement 2: {statement_2}")

        lines.append("\nA statement is a simple thought expressed by many comments, e.g., \"The video was well-edited.\" or \"I disagree with the premise of the video.\".")
        lines.append(f"First, think step by step about the two statements to determine if they agree. Finally, give your assessment of the agreement on a scale of {self.agreement_prompt_settings['min']} for total disagreement to {self.agreement_prompt_settings['max']} for total agreement. " \
                    f"The number {self.agreement_prompt_settings['neut']} is for unrelated statements (those which discuss different matters). Even if the sentiments of the statements are opposite: If they discuss different matters, the assessment should be {self.agreement_prompt_settings['neut']}. " \
                    "Provide your assessment in the form of JSON such as {\"agreement\": your_number_goes_here}.")

        prompt = "\n".join(lines)
        return prompt

    def _extract_statements(self, min_comments: int = 10):
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
    
    def _do_statements_agree(self, statement_1: str, statement_2: str, trials=3):
        # Make prompt
        prompt = self._build_prompt_do_statements_agree(statement_1, statement_2)

        # Send prompt to LLM
        for _ in range(trials):
            res_raw = self._llm.chat(prompt)
            rating = post_process_single_entry_json(res_raw)
            if rating is not None:
                break
        
        # If no rating could be extracted, mark the statements as being neutral
        rating_raw = rating
        if rating_raw is None:
            rating = self.agreement_prompt_settings['neut']

        # Save this snippet of information to a file
        save_snippet(
            {
                "statement_1": statement_1,
                "statement_2": statement_2,
                "video_title": self._video_title,
                "respose_raw": res_raw,
                "agreement_rating_raw": rating_raw,
                "agreement_rating": rating
            },
            self._data_dir_name
        )

        return rating
    
    def _check_agreement_all(self, comments_topk: List[Comment]):
        statement_scores = defaultdict(float)  # used to calculate a weighted score, e.g., 2.54 on a scale of -5 to 5
        statement_voices = defaultdict(dict)  # used to calculate how many people talk about the statement and what
        # their stand is

        # Get total like count to normalize scores
        total_likes = np.sum([comm.likes for comm in comments_topk])

        statements = sum(self._comment_statements.values(), [])  # get all statements, regardless of kind
        comparisons_all = list(itertools.product(statements, comments_topk))
        for statement, comment in tqdm(comparisons_all, desc="Measuring statement agreement with comments ..."):
            # Find out agreement between statement and comment
            rating = self._do_statements_agree(
                statement_1=statement,
                statement_2=comment.text
            )  # integer such as -5, -3, 0, 2, or 4 (from -5 to 5)

            # With the raw rating, we calculate two things: (1) a like-weighted score and (2) a tally of agreement or disagreement

            # Calculate agreement score = like-weighted rating
            likes_weight = comment.likes / total_likes
            score = rating * likes_weight

            # Add measured score to the statement's score
            statement_scores[statement] += score

            # --- (1) is done, now we do (2)

            # Find out the class of agreement: positive, neutral, disagreement
            voice_class = self._get_voice_class(rating)

            # So we can later measure how many people care about a topic and if they agree or disagree, add the comment's likes
            # to a tally of "votes".
            if voice_class not in statement_voices[statement]:
                statement_voices[statement][voice_class] = 0
            statement_voices[statement][voice_class] += likes_weight
        
        return statement_scores, statement_voices
    
    def _get_voice_class(self, agreement_rating: int):
        return self.voice_pos if agreement_rating > 0 else (self.voice_neg if agreement_rating < 0 else self.voice_neut)
    
    def run_analysis(self, limit_statements: Optional[int] = None, comment_top_k: int = 50):
        res = []

        # Extract statements
        self._comment_statements = self._extract_statements()

        # Number of statements can be limited for testing
        if limit_statements is not None:
            self._comment_statements = {
                kind: [statements[idx] for idx in np.random.choice(
                    np.arange(len(statements)), size=min(len(statements), 2), replace=False
                )]
                for (kind, statements) in self._comment_statements.items()
            }
        
        # Get the top k comments according to likes
        comments_topk = sorted(self._comments, key=lambda comm: comm.likes, reverse=True)[:comment_top_k]

        # Analyze their agreement to extracted statements
        statement_scores, statement_voices = self._check_agreement_all(comments_topk)

        # Print statement scores
        for statement, score in statement_scores.items():
            res.append(f"Score for statement '{statement}' -> {score:0.2f}")

        # Print fraction of agreement, disagreement, neutrality
        for statement, agree_info in statement_voices.items():
            # Remove neutral voices (but remember them)
            frac_neutral = agree_info.get(self.voice_neut, 0)
            frac_engaged = 1 - frac_neutral

            # Re-normalize fractions for other voices
            if frac_neutral > 0 and frac_engaged > 0:
                prob_mass = sum(agree_info.get(opinion, 0) for opinion in self.voices_nonneut)
                agree_info = {opinion: frac / prob_mass for (opinion, frac) in agree_info.items()}

            # Sort opinions alphabetically and keep only non-neutral opinions
            agree_info = sorted(agree_info.items(), key=lambda t: t[0])
            agree_info = [(opinion, fraction) for (opinion, fraction) in agree_info if opinion in self.voices_nonneut]

            # Format everything
            statement_str = f"Statement '{statement}'"
            if frac_engaged > 0:
                engagement_str = f"{100 * frac_engaged:0.2f}% are discussing this, out of those "
                opinion_str = ", ".join(f"{100 * fraction:0.0f}% {opinion}" for (opinion, fraction) in agree_info)
                discussion_str = engagement_str + opinion_str
            else:
                discussion_str = "No comments (of those checked) are discussing this."
            
            res.append(statement_str + f"->  " + discussion_str)

        return "\n".join(res)
