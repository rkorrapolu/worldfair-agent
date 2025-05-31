import json
import asyncio
from typing import TypedDict, List, Dict

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

import numpy as np
from openai import AsyncOpenAI
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

class GraphState(TypedDict):
  user_input: str
  responses: List[Dict[str, float]]

settings = Settings()
logger = initialize_logger(__name__, settings.is_development, settings.log_level)

def display_results(output: Dict):
  """Performance Visualization -> Structured Analytics Display"""
  console = Console()

  # Query Header
  query_panel = Panel(
      Text(output["query"], style="bold cyan"),
      title="[bold]Query Analysis[/bold]",
      border_style="blue"
  )
  console.print(query_panel)
  console.print()

  # Response Analysis Table
  response_table = Table(
      title="Response Generation Framework",
      show_header=True,
      header_style="bold magenta"
  )

  response_table.add_column("Variant", style="dim", width=12)
  response_table.add_column("Confidence", justify="center", style="green", width=10)
  response_table.add_column("Response Content", style="white", width=80)
  for resp in output["responses"]:
    confidence_color = "red" if resp["confidence"] < 0.5 else "yellow" if resp["confidence"] < 0.7 else "green"
    response_table.add_row(
      resp["variant"].upper(),
      f"[{confidence_color}]{resp['confidence']:.3f}[/{confidence_color}]",
      resp["response"][:150] + "..." if len(resp["response"]) > 150 else resp["response"]
    )

  console.print(response_table)
  console.print()
  # Performance Analytics Table
  analytics_table = Table(
    title="Performance Analytics Dashboard",
    show_header=True,
    header_style="bold yellow"
  )

  analytics_table.add_column("Metric", style="cyan", width=25)
  analytics_table.add_column("Value", justify="center", style="white", width=15)
  analytics_table.add_column("Performance Grade", justify="center", style="green", width=20)
  analytics = output["analytics"]

  # Performance grading framework
  avg_conf = analytics["avg_confidence"]
  grade = "EXCELLENT" if avg_conf > 0.8 else "GOOD" if avg_conf > 0.6 else "MODERATE"
  grade_color = "green" if avg_conf > 0.8 else "yellow" if avg_conf > 0.6 else "red"

  analytics_table.add_row("Average Confidence", f"{avg_conf:.3f}", f"[{grade_color}]{grade}[/{grade_color}]")
  analytics_table.add_row("Peak Confidence", f"{analytics['peak_confidence']:.3f}", "OPTIMAL")
  analytics_table.add_row("Response Diversity", f"{analytics['response_diversity']}", "HIGH")

  console.print(analytics_table)

async def generate_single_response(client: AsyncOpenAI, prompt: str, temperature: float, variant: str) -> Dict[str, float]:
  """Single Response Generation -> Confidence Scoring"""

  response = await client.chat.completions.create(
    model="gpt-4.1",
    messages=[{"role": "user", "content": prompt}],
    temperature=temperature,
    max_tokens=200
  )

  content = response.choices[0].message.content
  confidence_score = calculate_confidence(content, prompt)

  return {
    "response": content,
    "confidence": confidence_score,
    "variant": variant
  }

async def generate_responses_with_confidence(state: GraphState) -> GraphState:
  """User Input -> OpenAI Generation -> 3 Responses -> Confidence Scoring"""

  client = AsyncOpenAI(api_key=settings.genai_openai_api_key)
  user_query = state["user_input"]

  # Generate 3 diverse responses
  prompt_configs = [
    (f"Provide a comprehensive answer: {user_query}", 0.1, "analytical"),
    (f"Give a concise, practical response: {user_query}", 0.4, "practical"),
    (f"Offer an innovative perspective: {user_query}", 0.7, "creative")
  ]

  # Parallel execution
  tasks = [
    generate_single_response(client, prompt, temp, variant)
    for prompt, temp, variant in prompt_configs
  ]
  responses = await asyncio.gather(*tasks)
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
async def run_analysis(user_input: str) -> Dict:
  """Input Processing -> Response Generation -> Confidence Assessment"""
  graph = create_response_graph()
  # print(graph.get_graph().draw_mermaid())
  result = await graph.ainvoke({
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
      "peak_confidence": max(confidence_scores),
      "max_confidence": max(confidence_scores),
      "response_diversity": len(set(r["response"][:50] for r in result["responses"]))
    }
  }

async def main():
  output = await run_analysis("How can I improve my productivity at work?")
  display_results(output)

if __name__ == "__main__":
  asyncio.run(main())
