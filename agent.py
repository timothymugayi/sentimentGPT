import json
import logging
import os
import re
import openai

from typing import List

from config import get_openai_token_cost_for_model, SentimentSettings
from parser import SentimentOutputParser
from prompt import PROMPT
from rwschema import SentimentResponseSchema, SentimentRequest, SentimentResponse


logger = logging.getLogger(__name__)
settings = SentimentSettings()


class SentimentAgent(object):
    model_name: str = settings.llm_model_name
    min_word_count: int = 4
    openai.api_key = os.environ.get("OPENAI_API_KEY")

    @classmethod
    async def evaluate_sentiment(cls, statements: List[SentimentRequest]) -> SentimentResponse:
        """
        Evaluate the sentiment of a given statement using the OpenAI GPT-3 model.

        Args:
           cls (class): The class reference (conventionally named 'cls').
           statements (List): of text statement for sentiment analysis.

        Returns:
           SentimentResponseSchema: A dictionary-like object representing the sentiment analysis results.

        Raises:
           ValueError: If the 'statement' parameter is None.

        This function analyzes the sentiment of a given text statement using the OpenAI GPT-3 model.
        It calculates the sentiment, ranking, opinion, and profile of the statement.

        If the 'statement' is too short (contains fewer words than 'min_word_count'), the sentiment
        response will indicate 'short-text'.

        Usage:
           sentiment_response = await evaluate_sentiment(MyClass, "This is a sample statement.")
           logger.info(sentiment_response)
        """
        if not statements:
            raise ValueError("statements required")

        logger.info(statements)

        for statement in statements:
            words = re.findall(r'\b\w+\b', statement.message)
            word_count = len(words)
            if word_count <= cls.min_word_count:
                return SentimentResponse(
                    sentiments=[
                        SentimentResponseSchema(
                            sentiment="short-text",
                            ranking=-1,
                            opinion="short-text",
                            profile="short-text"
                        )
                    ]
                )

        prompt = PROMPT.format(statements=json.dumps([item.model_dump() for item in statements]))

        logger.info(prompt)

        response = openai.Completion.create(
            model=cls.model_name,
            prompt=prompt,
            temperature=0.7,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        usage = response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)
        total_cost: float = 0.0
        completion_cost = get_openai_token_cost_for_model(
            cls.model_name, completion_tokens, is_completion=True
        )
        prompt_cost = get_openai_token_cost_for_model(cls.model_name, prompt_tokens)
        total_cost += prompt_cost + completion_cost

        cost_info = (
            f"Tokens Used: {total_tokens}\n"
            f"\tPrompt Tokens: {prompt_tokens}\n"
            f"\tCompletion Tokens: {completion_tokens}\n"
            f"Total Cost (USD): ${total_cost}"
        )
        logger.info(cost_info)

        metadata = {
            "total_tokens": total_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_cost": total_cost
        }

        sentiments_output_parser = SentimentOutputParser(
            response_schemas=SentimentResponseSchema,
            text=response['choices'][0]['text'])

        sentiments = sentiments_output_parser.parse_json()

        return SentimentResponse(metadata=metadata, sentiments=sentiments)
