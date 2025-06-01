from typing import Dict, Any, List
from typing_extensions import TypedDict

from langgraph.graph import MessagesState


class UniversalEval(TypedDict):
    metric_key: str 
    metric_score: float
class GraphState(MessagesState):
  user_input: str
  responses: List[str]
  confidence_score: float
  confidence_threshold: float
  relevance_scores: List[float]
  contrast_score: float
  feedback_questions: Dict[str, Any]
  universal_eval: UniversalEval
