'''
EXTRA STEP: Clean up the extracted text 

'''

import json
import re
from pathlib import Path
from typing import Any

# Regex patterns

# Real control characters (keep \n \t \r)
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")

# Literal escaped control strings like "\u0000", "\u0001"
ESCAPED_CONTROL = re.compile(r"\\u000[0-9a-fA-F]")

# Hyphenation across line breaks: experi-\nments -> experiments
HYPHEN_LINEBREAK = re.compile(r"(\w)-\n(\w)")

# Whitespace cleanup
TOO_MANY_NEWLINES = re.compile(r"\n{3,}")
TRAILING_SPACES = re.compile(r"[ \t]+\n")
MULTI_SPACES = re.compile(r"[ \t]{2,}")


# Heuristic spacing fix (PDF word-joining)
def add_spaces(s: str) -> str:
    # Space after punctuation if followed immediately by a letter
    s = re.sub(r"([.,;:!?])([A-Za-z])", r"\1 \2", s)

    # Space between lowercase and uppercase letters
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)

    # Space between letters and digits
    s = re.sub(r"([A-Za-z])(\d)", r"\1 \2", s)
    s = re.sub(r"(\d)([A-Za-z])", r"\1 \2", s)

    # Collapse spaces again
    s = re.sub(r"[ \t]{2,}", " ", s)

    return s

# Core cleaning function
def clean_pdf_text(s: str) -> str:
    if not s:
        return s

    # Normalize newlines
    s = s.replace("\r\n", "\n").replace("\r", "\n")

    # Remove literal "\u0000" style artifacts
    s = ESCAPED_CONTROL.sub("", s)

    # Remove real control characters
    s = CONTROL_CHARS.sub("", s)

    # Fix hyphenation across line breaks
    s = HYPHEN_LINEBREAK.sub(r"\1\2", s)

    # Trim trailing spaces before newline
    s = TRAILING_SPACES.sub("\n", s)

    # Collapse repeated spaces
    s = MULTI_SPACES.sub(" ", s)

    # Collapse excessive blank lines (keep paragraphs)
    s = TOO_MANY_NEWLINES.sub("\n\n", s)

    # 🔹 Fix missing spaces caused by PDF extraction
    s = add_spaces(s)

    return s.strip()


# Recursive cleaner (in-place)

def clean_inplace(obj: Any) -> int:
    """Recursively clean any dict['text'] strings. Returns count cleaned."""
    count = 0

    if isinstance(obj, dict):
        if "text" in obj and isinstance(obj["text"], str):
            obj["text"] = clean_pdf_text(obj["text"])
            count += 1
        for v in obj.values():
            count += clean_inplace(v)

    elif isinstance(obj, list):
        for item in obj:
            count += clean_inplace(item)

    return count


# Main

def main(in_path: str, out_path: str):
    data = json.loads(Path(in_path).read_text(encoding="utf-8"))

    n = clean_inplace(data)

    Path(out_path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"Wrote cleaned JSON to: {out_path} (cleaned {n} text fields)")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python clean_text.py input.json output.json")
    main(sys.argv[1], sys.argv[2])
