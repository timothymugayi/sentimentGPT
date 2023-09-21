import logging
from typing import Union, Tuple, Dict, List

import tiktoken
from pydantic_settings import BaseSettings


logger = logging.getLogger(__name__)

# Source langchain
MODEL_COST_PER_1K_TOKENS = {
    # GPT-4 input
    "gpt-4": 0.03,
    "gpt-4-0314": 0.03,
    "gpt-4-0613": 0.03,
    "gpt-4-32k": 0.06,
    "gpt-4-32k-0314": 0.06,
    "gpt-4-32k-0613": 0.06,
    # GPT-4 output
    "gpt-4-completion": 0.06,
    "gpt-4-0314-completion": 0.06,
    "gpt-4-0613-completion": 0.06,
    "gpt-4-32k-completion": 0.12,
    "gpt-4-32k-0314-completion": 0.12,
    "gpt-4-32k-0613-completion": 0.12,
    # GPT-3.5 input
    "gpt-3.5-turbo": 0.0015,
    "gpt-3.5-turbo-0301": 0.0015,
    "gpt-3.5-turbo-0613": 0.0015,
    "gpt-3.5-turbo-16k": 0.003,
    "gpt-3.5-turbo-16k-0613": 0.003,
    # GPT-3.5 output
    "gpt-3.5-turbo-completion": 0.002,
    "gpt-3.5-turbo-0301-completion": 0.002,
    "gpt-3.5-turbo-0613-completion": 0.002,
    "gpt-3.5-turbo-16k-completion": 0.004,
    "gpt-3.5-turbo-16k-0613-completion": 0.004,
    # Others
    "gpt-35-turbo": 0.002,  # Azure OpenAI version of ChatGPT
    "text-ada-001": 0.0004,
    "ada": 0.0004,
    "text-babbage-001": 0.0005,
    "babbage": 0.0005,
    "text-curie-001": 0.002,
    "curie": 0.002,
    "text-davinci-003": 0.02,
    "text-davinci-002": 0.02,
    "code-davinci-002": 0.02,
    "ada-finetuned": 0.0016,
    "babbage-finetuned": 0.0024,
    "curie-finetuned": 0.012,
    "davinci-finetuned": 0.12,
}


def num_tokens_from_text(
    text: str, model: str = "gpt-3.5-turbo-0613", return_tokens_per_name_and_message: bool = False
) -> Union[int, Tuple[int, int, int]]:
    """Return the number of tokens used by a text."""
    # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        logger.debug("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model or "gpt-35-turbo" in model:
        logger.warning("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_text(
            text,
            model="gpt-3.5-turbo-0613",
            return_tokens_per_name_and_message=return_tokens_per_name_and_message
        )
    elif "gpt-4" in model:
        logger.warning("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_text(
            text,
            model="gpt-4-0613",
            return_tokens_per_name_and_message=return_tokens_per_name_and_message
        )
    else:
        raise NotImplementedError(
            f"""num_tokens_from_text() is not implemented for model {model}. See """
            f"""https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are """
            f"""converted to tokens."""
        )
    if return_tokens_per_name_and_message:
        return len(encoding.encode(text)), tokens_per_message, tokens_per_name
    else:
        return len(encoding.encode(text))


def num_tokens_from_messages(messages: List[Dict], model: str = "gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    global tokens_per_message
    num_tokens = 0
    for message in messages:
        for key, value in message.items():
            _num_tokens, tokens_per_message, tokens_per_name = num_tokens_from_text(
                value, model=model, return_tokens_per_name_and_message=True
            )
            num_tokens += _num_tokens
            if key == "name":
                num_tokens += tokens_per_name
        num_tokens += tokens_per_message
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def standardize_model_name(
    model_name: str,
    is_completion: bool = False,
) -> str:
    """
    Standardize the model name to a format that can be used in the OpenAI API.
    Args:
        model_name: Model name to standardize.
        is_completion: Whether the model is used for completion or not.
            Defaults to False.

    Returns:
        Standardized model name.

    """
    model_name = model_name.lower()
    if "ft-" in model_name:
        return model_name.split(":")[0] + "-finetuned"
    elif is_completion and (
        model_name.startswith("gpt-4") or model_name.startswith("gpt-3.5")
    ):
        return model_name + "-completion"
    else:
        return model_name


# source Langchain
def get_openai_token_cost_for_model(
    model_name: str, num_tokens: int, is_completion: bool = False
) -> float:
    """
    Get the cost in USD for a given model and number of tokens.

    Args:
        model_name: Name of the model
        num_tokens: Number of tokens.
        is_completion: Whether the model is used for completion or not.
            Defaults to False.

    Returns:
        Cost in USD.
    """
    model_name = standardize_model_name(model_name, is_completion=is_completion)
    if model_name not in MODEL_COST_PER_1K_TOKENS:
        raise ValueError(
            f"Unknown model: {model_name}. Please provide a valid OpenAI model name."
            "Known models are: " + ", ".join(MODEL_COST_PER_1K_TOKENS.keys())
        )
    return MODEL_COST_PER_1K_TOKENS[model_name] * (num_tokens / 1000)


def openaipricing(nb_tokens_used: int, model_name: str = None, embeddings=False, chatgpt=False) -> float:
    """
    Return cost per token in dollars for either GPT models or embeddings
    Args:
        nb_tokens_used: total token used by model
        model_name: llm model
        chatgpt: toggle gpt model compute cost
        embeddings: toggle embedding to compute tokens used cost

    Returns: cost
    """
    if not isinstance(nb_tokens_used, int):
        raise TypeError(
            f"nb_tokens_used Expected object of type int, got: {type(model_name)}"
        )
    if model_name and not isinstance(model_name, str):
        raise TypeError(
            f"model_name Expected object of type str, got: {type(model_name)}"
        )
    if chatgpt:
        #  GPT 4 has 2 prices
        #  [REQUEST] Prompt tokens: 0.03 USD per 1000 tokens are the parts of words fed into GPT-4
        #  [RESPONSE] while completion tokens are the content generated by GPT-4 at 0.06 USD per 1000 tokens
        cost = MODEL_COST_PER_1K_TOKENS.get(model_name, 1)
        return (cost / 1000.) * nb_tokens_used
    elif embeddings:
        # text-embedding-ada-002
        return (0.0004 / 1000.) * nb_tokens_used
    else:
        raise ValueError("Invalid option")


class SentimentSettings(BaseSettings):
    debug: bool = True
    llm_model_name: str = "text-davinci-003"

    class Config:
        env_prefix = 'SENTIMENT_'
