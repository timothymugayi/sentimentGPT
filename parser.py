import json
import logging
import re

from typing import Any, List

from pydantic import BaseModel, ValidationError, TypeAdapter

from rwschema import SentimentResponseSchema


logger = logging.getLogger(__name__)


def parse_json_markdown(json_string: str) -> dict:
    """
    Parse a JSON string from a Markdown string.

    Args:
        json_string: The Markdown string.

    Returns:
        The parsed JSON object as a Python dictionary.
    """
    # Try to find JSON string within triple backticks
    match = re.search(r"```(json)?(.*?)```", json_string, re.DOTALL)

    # If no match found, assume the entire string is a JSON string
    if match is None:
        json_str = json_string
    else:
        # If match found, use the content within the backticks
        json_str = match.group(2)

    # Strip whitespace and newlines from the start and end
    json_str = json_str.strip()

    # Parse the JSON string into a Python dictionary
    parsed = json.loads(json_str)

    return parsed


def parse_and_check_json_markdown(text: str) -> dict:
    """
    Parse a JSON string from a Markdown string and check that it
    contains the expected keys.

    Args:
        text: The Markdown string.

    Returns:
        The parsed JSON object as a Python dictionary.
    """
    try:
        json_obj = parse_json_markdown(text)
    except json.JSONDecodeError as e:
        raise OutputParserException(f"Got invalid JSON object. Error: {e}")
    return json_obj


class OutputParserException(ValueError):
    def __init__(self, error: Any):
        super(OutputParserException, self).__init__(error)


class SentimentOutputParser(BaseModel):
    response_schemas: Any
    text: str

    def parse_json(self) -> List[SentimentResponseSchema]:
        try:
            logger.debug(self.text.strip())
            json_obj = parse_and_check_json_markdown(self.text.strip())
            print(json_obj)
        except ValidationError as e:
            raise OutputParserException(f"Got invalid JSON object. Error: {e}")
        expected_keys = [_field for _field in list(SentimentResponseSchema.__annotations__)]
        for key in expected_keys:
            if key == "request_id":
                continue
            for item in json_obj:
                if key not in item:
                    raise OutputParserException(
                        f"Got invalid return object. Expected key `{key}` "
                        f"to be present, but got {json_obj}"
                    )

        ta = TypeAdapter(List[SentimentResponseSchema])

        return ta.validate_python(json_obj)
