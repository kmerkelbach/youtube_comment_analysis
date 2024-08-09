import re
import json
from typing import List, Dict


def split_text_if_long(text, max_len=1500):
    if len(text) <= max_len:
        return [text]
    else:
        # Collect words until we have reached maximum part length
        parts = []
        while len(text) > 0:
            # Select full words until hitting just below the maximum part length
            p = text[:max_len]
            p = " ".join(p.split()[:-1])

            # In case of a very long word, stop caring about word boundaries
            if len(p) == 0:
                p = text[:max_len]

            parts.append(p)
            text = text[len(p):].strip()
        return parts


def truncate_line(text: str, max_length: int) -> str:
    if len(text) > max_length:
        text = text[:max_length]
        text = " ".join(text.split()[:-1])
        text += "..."
    return text


def format_large_number(num):
    if abs(num) >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif abs(num) >= 1_000:
        return f"{num / 1_000:.1f}k"
    else:
        return str(num)


def post_process_extract_statements(raw: str) -> List[str]:
    # Split by lines
    lines = raw.split("\n")
    lines = [l for l in lines if len(l) > 0]  # remove blank lines

    # Look for enumeration at the start of the line (keep lines such as those starting with "4. ", "15.", "- ", or "• ")
    matched = [(l, re.search("^(\d+\.|-|•|\*)", l)) for l in lines]
    matched = [(l, m) for (l, m) in matched if m is not None]

    # Remove enumeration at the start of the line
    lines = [l[m.span()[-1]:] for (l, m) in matched]

    # Strip lines
    lines = [l.strip() for l in lines]
    
    return lines


def post_process_single_entry_json(raw):
    # Split into lines
    lines = raw.split("\n")

    # Beginning from the bottom, try to read JSON using a regex
    pattern = r'\{.*?\}'
    jso = None
    for line in reversed(lines):
        # Match regex
        matches = re.findall(pattern, line)
        
        if len(matches) == 0:
            continue

        # Parse json
        for m in reversed(matches):
            try:
                jso = json.loads(m)
                break
            except:
                pass

        if jso is not None:
            break

    # Abort if we did not find JSON
    if jso is None:
        return None

    # Extract only key in json
    if len(jso) > 1:
        return None  # invalid json
    val = list(jso.values())[0]
    
    return val