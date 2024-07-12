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