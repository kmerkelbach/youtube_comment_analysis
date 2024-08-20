import os
from typing import List, Dict
from json_tricks import dump, load
import pandas as pd
import numpy as np
from glob import glob


import logging
logger = logging.getLogger(__name__)


def get_root_dir():
    # Get the current file's directory
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    
    # Get the parent directory
    parent_directory = os.path.dirname(current_file_directory)
    
    return parent_directory


def make_dirs(path):
    os.makedirs(path, exist_ok=True)


def named_dir(name):
    directory = os.path.join(get_root_dir(), "data", name)
    make_dirs(directory)
    return directory


def get_line_count(path):
    with open(path, "rb") as f:
        num_lines = sum(1 for _ in f)
    return num_lines


def save_snippet(sni: dict, name: str, max_entries_per_file: int = 1000):
    # Make a DataFrame with the new piece of information
    df = pd.DataFrame([sni])

    # Make destination directory
    directory = named_dir(name)

    # Inspect already present files
    found_files = sorted(glob(os.path.join(directory, "*.csv")))
    if len(found_files) > 0:
        # Find out ID of last file
        last_path = found_files[-1]
        last_filename = os.path.split(last_path)[-1]
        last_filename = os.path.splitext(last_filename)[0]
        file_id = int(last_filename.split("_")[-1])

        # Find out its number of entries
        num_entries = get_line_count(last_path) - 1
        if num_entries >= max_entries_per_file:
            # Create a new file
            file_id += 1
        else:
            # Load the file and append to it
            df_old = pd.read_csv(last_path)
            df = pd.concat([df_old, df])
    else:
        file_id = 0

    # Save file
    filename = f"{name}_{file_id:06d}.csv"
    df.to_csv(os.path.join(directory, filename), index=False)


def load_json(file_path):
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                return load(file)
    except Exception as e:
        logger.error(f"Error reading JSON from {file_path}: {e}")
    return None


def save_json(file_path, data):
    try:
        with open(file_path, 'w') as file:
            dump(
                data,
                file,
                indent=4
            )
            logger.info(f"Wrote JSON file to {file_path}.")
            return True
    except IOError as e:
        logger.error(f"Error writing JSON to {file_path}: {e}")
    return False