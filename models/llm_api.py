import os
import time
from fireworks.client import Fireworks


import logging
logger = logging.getLogger(__name__)


class LLM:
    # Make this a singleton
    def __new__(cls):
        if not hasattr(cls, "_instance"):
            logger.info("Instantiating LLM.")
            cls._instance = super(LLM, cls).__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not hasattr(self, '_initialized'):
            self._initialized = True
            
            # Set up client
            self._client = Fireworks(api_key=os.getenv("FIREWORKSAI_KEY"))

            # Settings
            self.timeout = 5
            self.num_retries = 2

            # Statistics
            self._num_input_chars = 0
            self._num_output_chars = 0

    def get_num_input_chars(self):
        return self._num_input_chars

    def get_num_output_chars(self):
        return self._num_output_chars

    def _send_prompt(self, prompt):
        self._num_input_chars += len(prompt)
        chat_completion = self._client.chat.completions.create(
            model="accounts/fireworks/models/llama-v3-70b-instruct",
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}],
            temperature=0.7
        )

        res = chat_completion.choices[0].message.content
        self._num_output_chars += len(res)

        return res

    def chat(self, prompt):
        for _ in range(self.num_retries):
            try:
                return self._send_prompt(prompt)
            except:
                # Wait a little
                time.sleep(self.timeout)
        return None