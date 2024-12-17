import json
from typing import Any, Type, TypeVar, Union
from xml.sax.saxutils import escape as xml_escape

from pydantic import BaseModel, TypeAdapter

T = TypeVar("T")


def _convert_to_xml(obj, prefix=None, indent=0) -> str:
    """
    Serialize an object to a JSON string.

    This function first normalizes the object using _normalize(),
    then serializes it to a JSON string with indentation.

    Args:
        obj: The object to serialize.

    Returns:
        str: The JSON string representation of the object.
    """
    if prefix is None:
        prefix = []

    indent_str = "  " * indent
    if isinstance(obj, dict):
        result = []
        for key, value in obj.items():
            if not isinstance(value, (dict, list, set)):
                single_item = True
            else:
                single_item = len(value) == 1

            if single_item and not isinstance(value, (dict, list, set)):
                value_str = _convert_to_xml(value, prefix + [key], 0)
                closing_indent = ""
                newline = ""
            else:
                value_str = _convert_to_xml(value, prefix + [key], indent + 1)
                closing_indent = indent_str
                newline = "\n"

            if value_str.strip():
                result.append(
                    (
                        f"{indent_str}<{'.'.join((prefix + [key]))}>{newline}"
                        f"{value_str}{newline}"
                        f"{closing_indent}</{'.'.join((prefix + [key]))}>"
                    )
                )
            else:
                result.append((f"{indent_str}<{prefix}.{key}></{prefix}.{key}>"))

        return "\n".join(result)
    elif isinstance(obj, (list, set)):
        result = []
        for i, value in enumerate(obj):
            if not isinstance(value, (dict, list, set)):
                single_item = True
            else:
                single_item = len(value) == 1

            if single_item and not isinstance(value, (dict, list, set)):
                value_str = _convert_to_xml(value, prefix, 0)
                closing_indent = ""
                newline = ""
            else:
                value_str = _convert_to_xml(value, prefix, indent + 1)
                closing_indent = indent_str
                newline = "\n"

            if value_str.strip():
                result.append(
                    (
                        f"{indent_str}<{'.'.join((prefix + [str(i)]))}>{newline}"
                        f"{value_str}{newline}"
                        f"{closing_indent}</{'.'.join((prefix + [str(i)]))}>"
                    )
                )
            else:
                result.append(
                    (
                        f"{indent_str}<{'.'.join((prefix + [str(i)]))}></{'.'.join((prefix + [str(i)]))}>"
                    )
                )
        return "\n".join(result)
    else:
        return indent_str + xml_escape(str(obj))


async def compile_user_prompt(**kwargs) -> str:
    """
    Compile a user prompt from keyword arguments.

    Each keyword argument is serialized and wrapped in XML-like tags.

    Args:
        **kwargs: Keyword arguments to include in the prompt.

    Returns:
        str: The compiled user prompt.
    """
    prompt_pieces = []
    for key, value in kwargs.items():
        value = _convert_to_xml(await _convert_to_dict(value), None, 1)
        prompt_pieces.append(f"<{key}>\n{value}\n</{key}>")

    return "\n\n".join(prompt_pieces)


def parse_obj_response(response_model: Type[T], content: str) -> T:
    """
    Parse an object response from the LLM.

    This function extracts JSON from a code block in the response content
    and constructs an instance of the response model from it.

    Args:
        response_model (BaseModel): The expected response model class.
        content (str): The response content from the LLM.

    Returns:
        An instance of the response model.
    """
    if "```json" in content:
        json_start = content.find("```json") + 7
    elif "```" in content:
        json_start = content.find("```") + 3

    json_end = content.find("```", json_start)
    cleaned_json = clean_json_string(content[json_start:json_end].strip())
    obj = json.loads(cleaned_json)

    try:
        if issubclass(response_model, BaseModel):
            return response_model(**obj)
    except TypeError:
        pass

    return TypeAdapter(response_model).validate_python(obj)


def parse_block_response(block_type: str, content: str) -> str:
    """
    Parse a block response from the LLM.

    This function extracts the content from a code block of the specified type
    in the response content.

    Args:
        block_type (str): The type of block to extract (e.g., "python", "sql").
        content (str): The response content from the LLM.

    Returns:
        str: The extracted content from the code block.
    """
    if "```" not in content:
        return content

    if f"```{block_type}" in content:
        start = content.find(f"```{block_type}") + 3 + len(block_type)
    elif "```" in content:
        start = content.find("```") + 3

    end = content.find("```", start)
    return content[start:end].strip()


async def _convert_to_dict(obj: Any) -> dict:
    """
    Recursively normalize an object for serialization.

    This function handles Pydantic BaseModel instances, dictionaries, and lists.
    Other types are returned as-is.

    Args:
        obj: The object to normalize.

    Returns:
        The normalized version of the object.
    """
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    elif isinstance(obj, dict):
        return {k: await _convert_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [await _convert_to_dict(v) for v in obj]
    elif isinstance(obj, (int, float, str, bool)):
        return obj
    else:
        return await _convert_to_dict(await obj.json() if obj is not None else "None")


def clean_json_string(json_str: str) -> str:
    """
    Clean a JSON string by properly handling newlines within quoted values.

    Args:
        json_str (str): The potentially invalid JSON string to clean

    Returns:
        str: A cleaned JSON string with escaped newlines
    """
    # State variables
    in_quotes = False
    is_escaped = False
    clean_chars = []

    for char in json_str:
        if char == '"' and not is_escaped:
            in_quotes = not in_quotes

        elif char == "\\":
            is_escaped = True
            clean_chars.append(char)
            continue

        # Handle newlines inside quoted strings
        elif char in "\n\r" and in_quotes:
            clean_chars.append("\\n")
            if char == "\r" and json_str[json_str.index(char) + 1] == "\n":
                continue
            continue

        is_escaped = False
        clean_chars.append(char)

    return "".join(clean_chars)
