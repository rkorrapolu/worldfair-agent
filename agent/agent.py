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

  response_table.add_column("Index", style="dim", width=8)
  response_table.add_column("Relevance", justify="center", style="green", width=10)
  response_table.add_column("Response Content", style="white", width=80)

  responses = output.get("responses", [])
  relevance_scores = output["analytics"].get("relevance_scores", [])

  for i, resp in enumerate(responses):
    relevance = relevance_scores[i] if i < len(relevance_scores) else 0.0
    relevance_color = "red" if relevance < 0.5 else "yellow" if relevance < 0.8 else "green"
    content = ""
    if isinstance(resp, dict):
      content = resp.get("response", resp.get("content", str(resp)))
    else:
      content = str(resp)

    response_table.add_row(
        f"#{i + 1}",
        f"[{relevance_color}]{relevance:.3f}[/{relevance_color}]",
        content[:150] + "..." if len(content) > 150 else content
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
  conf_score = analytics["confidence_score"]
  grade = "EXCELLENT" if conf_score > 0.8 else "GOOD" if conf_score > 0.6 else "MODERATE"
  grade_color = "green" if conf_score > 0.8 else "yellow" if conf_score > 0.6 else "red"

  analytics_table.add_row("Confidence Score", f"{conf_score:.3f}", f"[{grade_color}]{grade}[/{grade_color}]")
  analytics_table.add_row("Average Relevance", f"{analytics['avg_relevance']:.3f}", "OPTIMAL")
  analytics_table.add_row("Peak Relevance", f"{analytics['peak_relevance']:.3f}", "MAXIMUM")
  analytics_table.add_row("Contrast Score", f"{analytics['contrast_score']:.3f}", "DIVERSITY")

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
def create_response_graph(checkpointer):
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

  return workflow.compile(checkpointer=checkpointer)

# Execution Framework
async def run_analysis(user_input: str) -> Dict:
  """Input Processing -> Response Generation -> Confidence Assessment"""
  graph = create_response_graph(None)
  # print(graph.get_graph().draw_mermaid())
  result = await graph.ainvoke({
    "user_input": user_input,
    "responses": []
  })

  # Performance metrics
  confidence_score = result.get("confidence_score", 0.0)
  relevance_scores = result.get("relevance_scores", [])
  contrast_score = result.get("contrast_score", 0.0)
  return {
    "query": user_input,
    "responses": result["responses"],
    "analytics": {
      "confidence_score": round(confidence_score, 3),
      "relevance_scores": relevance_scores,
      "contrast_score": round(contrast_score, 3),
      "avg_relevance": round(np.mean(relevance_scores), 3) if relevance_scores else 0.0,
      "peak_relevance": max(relevance_scores) if relevance_scores else 0.0,
    }
  }

async def main():
  output = await run_analysis("Where is the location of 2025 AI Engineer World's Fair Agents Hackathon in SF?")
  display_results(output)

if __name__ == "__main__":
  asyncio.run(main())
