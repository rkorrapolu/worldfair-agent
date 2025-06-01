from typing import List
from langgraph.graph import MessagesState

class GraphState(MessagesState):
  user_input: str
  responses: List[str]
  confidence_score: float
  relevance_scores: List[float]
  contrast_score: float