from typing import Optional, Dict, List
import os
from dateutil import parser as dateparser
import numpy as np
import googleapiclient.discovery
from tqdm import tqdm


from structures.comment import Comment


import logging
logger = logging.getLogger(__name__)


# Fields for navigating API responses
field_npt = "nextPageToken"
field_tlc = "topLevelComment"
field_sni = "snippet"
field_id = "id"
field_items = "items"
field_textDis = "textDisplay"
field_repl = "replies"
field_repl_count = "totalReplyCount"
field_comments = "comments"
field_authorDis = "authorDisplayName"
field_likes = "likeCount"
field_publAt = "publishedAt"
field_title = "title"
field_channel_title = "channelTitle"
field_stats = "statistics"
field_cmmt_cnt = "commentCount"
field_audio_lang = "defaultAudioLanguage"


class YoutubeAPI:
    def __init__(self) -> None:
        # Set up client
        api_key = os.getenv("YOUTUBE_API_KEY")
        self._client = googleapiclient.discovery.build(
            "youtube",
            "v3",
            developerKey=api_key
        )

        # Allow user to select a video ID
        self._current_video_id = None

        # Cache video info
        self._video_info = {}

    def set_current_video(self, video_id: str):
        self._current_video_id = video_id

    def reset_current_video(self):
        self._current_video_id = None

    def _get_video_id(self, video_id_user_provided: Optional[str]) -> str:
        ids = [video_id_user_provided, self._current_video_id]

        for potential_id in ids:
            if potential_id is not None:
                return potential_id
            
        raise ValueError("No YouTube video ID specified!")
    
    def get_video_info(self, video_id: Optional[str] = None) -> Dict:
        video_id = self._get_video_id(video_id)

        # Try to retrieve from cache
        if video_id in self._video_info:
            return self._video_info[video_id]

        # Retrieve info
        request = self._client.videos().list(
            part="snippet,contentDetails,statistics",
            id=video_id
        )
        response = request.execute()

        # Save in cache
        self._video_info[video_id] = response

        return response
    
    def get_title(self, video_id: Optional[str] = None, unk: str = "<Unknown Title>") -> str:
        video_id = self._get_video_id(video_id)

        video_info = self.get_video_info(video_id)

        try:
            return video_info[field_items][0][field_sni][field_title]
        except:
            return unk
        
    def get_comment_count(self, video_id: Optional[str] = None, unk: float = float('nan')) -> int:
        video_id = self._get_video_id(video_id)

        video_info = self.get_video_info(video_id)

        try:
            return int(video_info[field_items][0][field_stats][field_cmmt_cnt])
        except:
            return unk
        
    def get_creator_name(self, video_id: Optional[str] = None, unk: str = "<Unknown Creator>") -> str:
        video_id = self._get_video_id(video_id)

        video_info = self.get_video_info(video_id)

        try:
            return video_info[field_items][0][field_sni][field_channel_title]
        except:
            return unk
    
    def _get_comments_page_raw(self, video_id: Optional[str] = None, max_results: int = 50, page_token: str = None):
        video_id = self._get_video_id(video_id)

        request = self._client.commentThreads().list(
            part="snippet,replies",
            order="time",
            videoId=video_id,
            maxResults=max_results,
            textFormat="plainText",
            pageToken=page_token
        )
        response = request.execute()
        return response
    
    def _get_comments_raw(self, video_id: Optional[str] = None, req: int = 200, max_count_per_page: int = 100):
        video_id = self._get_video_id(video_id)

        # Find out the number of comments. We will use it to estimate the number of pages required
        total_comment_count = self.get_comment_count(video_id)
        num_pages_req_max = int(np.ceil(total_comment_count / max_count_per_page))
        
        comments = []
        if req is None:
            req = 2**63 - 1
        comments_left = req
        have_enough_comments = False
        next_page_token = None

        logger.info("Starting raw comment retrieval.")

        page_idx = 0
        while not have_enough_comments:
            new_comments_dict = self._get_comments_page_raw(video_id, max_results=min(max_count_per_page, comments_left), page_token=next_page_token)
            next_page_token = new_comments_dict.get(field_npt, None)

            new_comments = new_comments_dict["items"]
            comments += new_comments
            logger.info(f"Received {len(new_comments)} top-level comments.")

            comments_left -= len(new_comments)

            if next_page_token is None:
                break
            page_idx += 1
            logger.info(f"Requesting another page (page {page_idx + 1} of at most {num_pages_req_max}) ...")

            have_enough_comments = comments_left <= 0

        logger.info(f"Finished raw comment retrieval of {len(comments)} top-level comments.")
        return comments
    
    # Find comments for which we don't have all of the replies
    def _retrieve_all_replies(self, comments_raw):
        for comm in tqdm(comments_raw, desc="Getting replies for comments with missing replies ..."):
            # Skip if the required fields are not present
            if field_sni not in comm:
                continue
        
            # Check if we have all of the replies
            num_replies_stated = comm[field_sni][field_repl_count]
            
            if field_repl not in comm:
                num_replies_actual = 0
            else:
                num_replies_actual = len(comm[field_repl][field_comments])
        
            # Skip if we have all of the replies
            if num_replies_stated == num_replies_actual:
                continue
        
            # Get all replies for this comment
            replies = []
            next_page_token = None
            while len(replies) < num_replies_stated:
                # Request replies
                req_kwargs = dict(
                    part="snippet",
                    parentId=comm[field_id],
                    textFormat="plainText",
                    maxResults=100
                )
                if next_page_token is not None:
                    req_kwargs["pageToken"] = next_page_token
                request = self._client.comments().list(**req_kwargs)
                response = request.execute()
        
                # Store replies
                replies += response[field_items]

                # Go to next page
                next_page_token = response.get(field_npt, None)
        
            # Save replies back to the original dictionary, preserving its structure
            comm[field_repl][field_comments] = replies
    
    def _instantiate_comment(self, comm):
        # Get text
        sni = comm[field_sni]
        sni = sni.get(field_tlc, sni)
        sni = sni.get(field_sni, sni)
        text = sni[field_textDis]

        # Get author
        author = sni[field_authorDis]

        # Get likes
        likes = sni[field_likes]

        # Get replies
        if field_repl in comm:
            replies = comm[field_repl][field_comments]
            replies = [self._instantiate_comment(r) for r in replies]
        else:
            replies = []

        # Get time of publishing
        t_published = dateparser.parse(sni[field_publAt])

        # Create comment object
        comment = Comment(
            author=author,
            text=text,
            time=t_published,
            likes=likes,
            replies=replies
        )
        
        return comment
    
    def _instantiate_all_comments(self, comments_raw):
        comments_new = []
        for comm in tqdm(comments_raw, desc="Converting comments to our own class ..."):
            comments_new.append(self._instantiate_comment(comm))
        return comments_new

    @staticmethod
    def _key_for_comment(comm: Comment) -> str:
        return f"{comm.author}@{comm.time.isoformat()}: '{comm.text}' ({len(comm.replies)} replies)"
    
    # Deduplicate top-level comments
    def _deduplicate_toplevel(self, comments: List[Comment]):
        seen = set()
        deduped = []
        
        for comm in tqdm(comments, desc="Deduplicating comments ..."):
            # Get unique key for comment
            k = self._key_for_comment(comm)
        
            # Skip comment if we have already seen it
            if k in seen:
                continue
            
            seen.add(k)
        
            # Add comment to list of deduplicated comments
            deduped.append(comm)

        return deduped
    
    def get_comments(self, video_id: Optional[str] = None, req: int = None):
        video_id = self._get_video_id(video_id)

        print(f"Starting comments retrieval for video ID {video_id} ('{self.get_title(video_id)}')")

        # Raw comments in the form of a (JSON) dictionary
        comments_raw = self._get_comments_raw(video_id=video_id, req=req)

        # Not all of the comments have all their replies included. Let's retrieve them.
        self._retrieve_all_replies(comments_raw)
        # (this is an in-place operation)

        # Instance comments (which, so far, are dictionaries) as our own class
        comments = self._instantiate_all_comments(comments_raw)

        # Remove duplicate comments
        comments = self._deduplicate_toplevel(comments)
        
        return comments