from typing import List, TypedDict


class GraphState(TypedDict):
  user_input: str
  responses: List[str]
  confidence_score: float
  relevance_scores: List[float]
  contrast_score: float