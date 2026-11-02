"""Derive one portable output component from a text query."""
import re


def query_filename(text):
    cleaned = re.sub(r'[/\\\\:\x00-\x1f\x7f<>|"?*]', '_', text.strip()).strip(' .')
    if not cleaned or len(cleaned.encode("utf-8")) > 180:
        raise ValueError("Query text needs a usable output name (at most 180 UTF-8 bytes).")
    return cleaned
