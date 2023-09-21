import logging
import os
import uvicorn

from typing import List
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from openai.error import AuthenticationError
from starlette import status
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from agent import SentimentAgent
from config import num_tokens_from_messages, SentimentSettings, \
    get_openai_token_cost_for_model
from rwschema import SentimentRequest, SentimentTokenResponseSchema, SentimentResponse


logger = logging.getLogger(__name__)
settings = SentimentSettings()

load_dotenv()


if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = ""

app = FastAPI(
    title="SentimentGPT",
    description="SentimentGPT",
    summary="SentimentGPT text classification",
    version="0.0.1",
    terms_of_service="http://example.com/terms/",
    contact={
        "name": "Astro-X the Amazing",
        "url": "http://x-force.example.com/contact/",
        "email": "django.course@gmail.com",
    },
    license_info={
        "name": "Copyright",
        "url": "https://timothymugayi.com/licenses/LICENSE-2.0.html",
    }
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(AuthenticationError)
async def validation_exception_handler(request: Request, exc: AuthenticationError):
    logger.debug(request)
    return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content=jsonable_encoder({"detail": exc.user_message}),
    )


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/api/v1/sentiment/evaluate")
async def chat(messages: List[SentimentRequest]) -> SentimentResponse:
    return await SentimentAgent.evaluate_sentiment(messages)


@app.post("/api/v1/sentiment/evaluate_tokens")
async def chat(messages: List[SentimentRequest]) -> SentimentTokenResponseSchema:
    messages_as_dicts = [message.model_dump(exclude_none=True) for message in messages]
    num_tokens = num_tokens_from_messages(messages_as_dicts)
    total_cost = get_openai_token_cost_for_model(
        model_name=settings.llm_model_name, num_tokens=num_tokens)
    return SentimentTokenResponseSchema(
        total_messages=len(messages_as_dicts),
        num_tokens=num_tokens,
        total_cost=total_cost)


if __name__ == "__main__":
    uvicorn.run(host="0.0.0.0", port=8000, app=app)
