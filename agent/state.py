from typing import Dict, Any, List
from langgraph.graph import MessagesState

class GraphState(MessagesState):
  user_input: str
  responses: List[str]
  confidence_score: float
  relevance_scores: List[float]
  contrast_score: float
  feedback_questions: Dict[str, Any]
