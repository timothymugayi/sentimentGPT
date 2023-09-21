from typing import Dict, Any, Optional, List

from pydantic import BaseModel, Field


class SentimentTokenResponseSchema(BaseModel):
    total_messages: int
    num_tokens: int
    total_cost: float


class SentimentResponseSchema(BaseModel):
    request_id: Optional[str] = None
    sentiment: str = Field(...,
                           description="The sentiment associated with the response.")
    ranking: int = Field(...,
                         description="The ranking or score of the response. "
                                     "a numeric measure of the bullishness or postive / bearishness negative. "
                                     "This ranges from 1 (extreme negative sentiment) "
                                     "to 10 (extreme positive sentiment.")
    opinion: str = Field(...,
                         description="Model opinion or feedback.")
    profile: str = Field(...,
                         description="The user's profile classification")


class SentimentResponse(BaseModel):
    sentiments: List[SentimentResponseSchema]
    metadata: Optional[Dict[str, Any]] = Field({}, description="Request metadata")


class SentimentRequest(BaseModel):
    message: str
    request_id: str
