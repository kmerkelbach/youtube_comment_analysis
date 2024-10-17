# YouTube Comment Analysis

Features: Clustering into topics, sentiment analysis, LLM-driven summary of comment section

## Setup

1. Clone this repository:
`
git clone git@github.com:kmerkelbach/youtube_comment_analysis.git
`

2. Install requirements using pip:
```commandline
cd youtube_comment_analysis
pip install -r requirements.txt
```

3. Get a YouTube API key ([instructions](https://developers.google.com/youtube/v3/getting-started)) and store it in an environment variable `YOUTUBE_API_KEY` ([instructions for Ubuntu](https://help.ubuntu.com/community/EnvironmentVariables)).
4. Get a Fireworks AI ([https://fireworks.ai/](https://fireworks.ai/)) API key and store it in an environment variable `FIREWORKSAI_KEY`. This allows us easy access to an LLM.
5. You're done! Try one of the examples.

## Examples

### Individual Video
To analyze a single video, e.g., [this video by Mark Rober](https://www.youtube.com/watch?v=QpwJEYGCngI) titled "You've Never Seen A Wheelchair Like This", run the following script:
```commandline
python execution/run_comment_analysis.py --video_id="QpwJEYGCngI"
```
Note that the video ID is the part of the URL after `watch?v=` and before any `&` that might be contained in the URL.

This script will fetch all comments and all replies and analyze the sentiment, cluster comments based on topic and provide an LLM-generated summary of the analysis.

Here's the analysis summary for the Mark Rober video:

> The sentiment analysis of the comments reveals that 78.8% of comments are positive, 20.3% are negative, and 0.8% are neutral. The majority of comments are overwhelmingly positive, with many expressing heartfelt appreciation, energetic enthusiasm, and emotional reactions to the heartwarming video.<br><br>
The statements extracted from the comments show that most commenters agree that the wheelchair is amazing and impressive, Mark Rober is a kind and awesome person, the video is heartwarming and inspiring, and the kid and his parents are awesome and great. There are also comments praising Cash's independence and inspiration, as well as the creativity and innovation in the video.<br><br>
The clustering of topics reveals 11 distinct clusters, with the largest clusters being "Admiration and Appreciation" (19% of comments) and "Energetic Enthusiasm" (18% of comments). Other notable clusters include "Heartfelt Appreciation" (11% of comments), "Community Appreciation and Uplift" (14% of comments), and "Praise for Cash's Father and Family" (5% of comments).<br><br>
Overall, the comments are overwhelmingly positive, with many expressing admiration and appreciation for the video, Mark Rober, and the kid and his family.

### CSV File (Table)
You can also run the analysis for multiple videos in batch operation.
```commandline
python execution/run_csv_table.py --csv_path your_csv_file.csv
```
Your CSV file needs to have a field `URL` containing the YouTube URLs (not video IDs) as the first column.

Since the analysis results are written back to the CSV file every time a video finishes, you can resume aborted runs just by restarting.

## Contributing
You are welcome to contribute, though please consider that since this is a small personal project I developed in my free time, I will not be able to react to PRs quickly (or at all).
Feel free to fork this repository, too!

## Author
**Kilian Merkelbach**
- [Website](https://kmerkelbach.com/)
- [GitHub](https://github.com/kmerkelbach)
- [LinkedIn](https://www.linkedin.com/in/kilian-merkelbach-bb3575125/)
- [Twitter/X](https://x.com/kmerkelbach0)
