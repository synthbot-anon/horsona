import json
from typing import TextIO


def _load_json_from_string(data: str) -> dict:
    lines = data.split("\n")
    updated_lines = []
    for line in lines:
        in_str = False
        in_escape = False
        hash_start = None

        for i, char in enumerate(line):
            if char == '"' and not in_escape:
                in_str = not in_str
            elif char == "\\" and in_str:
                in_escape = not in_escape
            elif char == "#" and not in_str:
                hash_start = i
                break

        if hash_start is None:
            updated_lines.append(line)
            continue

        updated_lines.append(line[:hash_start])

    return json.loads("\n".join(updated_lines))


def load_json_with_comments(data: str | TextIO) -> dict:
    if isinstance(data, str):
        data_str = data
    else:
        data_str = data.read()

    return _load_json_from_string(data_str)
