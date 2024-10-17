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

