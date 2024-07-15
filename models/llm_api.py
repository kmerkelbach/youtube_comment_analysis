import os
import time
from fireworks.client import Fireworks


import logging
logger = logging.getLogger(__name__)


class LLM:
    # Make this a singleton
    def __new__(cls):
        if not hasattr(cls, "instance"):
            logger.info("Instantiating LLM.")
            cls.instance = super(LLM, cls).__new__(cls)
        return cls.instance

    def __init__(self) -> None:
        # Set up client
        self._client = Fireworks(api_key=os.getenv("FIREWORKSAI_KEY"))

        # Settings
        self.timeout = 5
        self.num_retries = 2

    def _send_prompt(self, prompt):
        chat_completion = self._client.chat.completions.create(
            model="accounts/fireworks/models/llama-v3-70b-instruct",
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}],
            temperature=0.7
        )
        return chat_completion.choices[0].message.content

    def chat(self, prompt):
        for _ in range(self.num_retries):
            try:
                return self._send_prompt(prompt)
            except:
                # Wait a little
                time.sleep(self.timeout)
        return None