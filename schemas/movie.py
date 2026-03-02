from pydantic import BaseModel
from typing import List, Optional

class MovieRecommendation(BaseModel):
    movie_id: int
    score: Optional[float] = None
    title: Optional[str] = None

class RecommendationResponse(BaseModel):
    user_id: Optional[int] = None
    movie_id: Optional[int] = None
    recommendations: List[MovieRecommendation]
    method: str