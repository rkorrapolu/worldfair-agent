import json
from typing import TypedDict, List, Dict

import numpy as np
from openai import OpenAI
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from langgraph.graph import StateGraph

from setup_logger import initialize_logger

# Configuration
class Settings(BaseSettings):
  model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

  app_name: str = "worldfair-agent"
  version: str = "0.1.0"
  host: str = "0.0.0.0"
  port: int = 7001

  # Environment configuration
  environment: str = Field(default="local", env="ENVIRONMENT")

  # Agent configuration
  genai_openai_api_key: str = Field(default="", env="OPENAI_API_KEY")

  # Logging
  log_level: str = "INFO"

  # Computed properties
  @property
  def is_development(self) -> bool:
    return self.environment.lower() == "local"

settings = Settings()
logger = initialize_logger(__name__, settings.is_development, settings.log_level)

class GraphState(TypedDict):
  user_input: str
  responses: List[Dict[str, float]]

def generate_responses_with_confidence(state: GraphState) -> GraphState:
  """User Input -> OpenAI Generation -> 3 Responses -> Confidence Scoring"""

  client = OpenAI(api_key=settings.genai_openai_api_key)
  user_query = state["user_input"]

  # Generate 3 diverse responses
  prompts = [
    f"Provide a comprehensive answer: {user_query}",
    f"Give a concise, practical response: {user_query}",
    f"Offer an innovative perspective: {user_query}"
  ]

  responses = []
  for i, prompt in enumerate(prompts):
    response = client.chat.completions.create(
      model="gpt-4.1",
      messages=[{"role": "user", "content": prompt}],
      temperature=0.1 + (i * 0.3),  # Vary creativity
      max_tokens=200
    )

    content = response.choices[0].message.content

    # Calculate confidence metrics
    confidence_score = calculate_confidence(content, user_query)
    responses.append({
      "response": content,
      "confidence": confidence_score,
      "variant": f"approach_{i + 1}"
    })

  return {"user_input": user_query, "responses": responses}

def calculate_confidence(response: str, query: str) -> float:
  """Response Analysis -> Confidence Metrics"""

  # Length relevance (0.3 weight)
  length_score = min(len(response) / 150, 1.0) * 0.3

  # Keyword overlap (0.4 weight)
  query_words = set(query.lower().split())
  response_words = set(response.lower().split())
  overlap_score = len(query_words.intersection(response_words)) / max(len(query_words), 1) * 0.4

  # Response completeness (0.3 weight)
  completeness_score = (1.0 if response.endswith(('.', '!', '?')) else 0.7) * 0.3

  return round(length_score + overlap_score + completeness_score, 3)

# Graph Construction
def create_response_graph():
  """Graph Assembly -> Single Node -> Multi-Response Generator"""
  workflow = StateGraph(GraphState)
  # Single processing node
  workflow.add_node("generate_responses", generate_responses_with_confidence)

  # Graph flow
  workflow.set_entry_point("generate_responses")
  workflow.set_finish_point("generate_responses")

  return workflow.compile()

# Execution Framework
def run_analysis(user_input: str) -> Dict:
  """Input Processing -> Response Generation -> Confidence Assessment"""
  graph = create_response_graph()
  result = graph.invoke({
    "user_input": user_input,
    "responses": []
  })

  # Performance metrics
  confidence_scores = [r["confidence"] for r in result["responses"]]
  return {
    "query": user_input,
    "responses": result["responses"],
    "analytics": {
      "avg_confidence": round(np.mean(confidence_scores), 3),
      "max_confidence": max(confidence_scores),
      "response_diversity": len(set(r["response"][:50] for r in result["responses"]))
    }
  }

if __name__ == "__main__":
  output = run_analysis("How can I improve my productivity at work?")
  print(json.dumps(output, indent=2))
