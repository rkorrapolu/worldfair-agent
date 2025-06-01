import asyncio
import json
import os
import asyncio
from typing import Dict

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

import numpy as np
from openai import AsyncOpenAI
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

def display_results(output: Dict):
  """Performance Visualization -> Structured Analytics Display"""
  console = Console()

  # Query Header
  query_panel = Panel(
      Text(output["query"], style="bold cyan"),
      title="[bold]Query Analysis[/bold]",
      border_style="blue"
  )
  console.print()
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
async def run_analysis(user_input: str) -> Dict:
  """Input Processing -> Response Generation -> Confidence Assessment"""
  graph = create_response_graph()
  # print(graph.get_graph().draw_mermaid())
  result = await graph.ainvoke({
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
