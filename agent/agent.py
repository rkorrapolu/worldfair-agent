import asyncio
import json
import os
from typing import Dict

import numpy as np
from openai import AsyncOpenAI, OpenAI
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from langgraph.graph import StateGraph

from state import GraphState
from setup_logger import initialize_logger
from confidence_evaluator import calculate_contrast, calculate_confidence, evaluate_relevance
from dotenv import load_dotenv

load_dotenv()

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

async def generate_single_response(client: AsyncOpenAI, prompt: str, temperature: float) -> str:
  """Single Response Generation -> Confidence Scoring"""

  response = await client.chat.completions.create(
    model="gpt-4.1",
    messages=[{"role": "user", "content": prompt}],
    temperature=temperature,
    max_tokens=200
  )

  content = response.choices[0].message.content

  return content or ""

async def generate_responses(state: GraphState) -> GraphState:
  """User Input -> OpenAI Generation -> 3 Responses -> Confidence Scoring"""

  client = AsyncOpenAI(api_key=settings.genai_openai_api_key or os.environ.get("OPENAI_API_KEY"))
  user_query = state["user_input"]

  # Generate 3 diverse responses
  prompt_configs = [
    (f"Provide a comprehensive answer: {user_query}", 0.1),
    (f"Give a concise, practical response: {user_query}", 0.4),
    (f"Offer an innovative perspective: {user_query}", 0.7)
  ]

  # Parallel execution
  tasks = [
    generate_single_response(client, prompt, temp)
    for prompt, temp in prompt_configs
  ]
  responses = await asyncio.gather(*tasks)
  state["responses"] = responses
  return state

# Graph Construction
def create_response_graph():
  """Graph Assembly -> Single Node -> Multi-Response Generator"""
  workflow = StateGraph(GraphState)
  # Single processing node
  workflow.add_node("generate_responses", generate_responses)
  workflow.add_node("evaluate_relevance", evaluate_relevance)
  workflow.add_node("calculate_contrast", calculate_contrast)
  workflow.add_node("calculate_confidence", calculate_confidence)

  # Graph flow
  workflow.set_entry_point("generate_responses")
  workflow.add_edge("generate_responses", "evaluate_relevance")
  workflow.add_edge("evaluate_relevance", "calculate_contrast")
  workflow.add_edge("calculate_contrast", "calculate_confidence")
  workflow.set_finish_point("calculate_confidence")

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
  confidence_scores = [r["confidence_score"] for r in result["responses"]]
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
