import asyncio
import json
import os
import asyncio
from typing import cast, Dict

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from jinja2 import Environment, FileSystemLoader
from dotenv import load_dotenv

import numpy as np
from openai import AsyncOpenAI
from state import GraphState
from setup_logger import initialize_logger
from langgraph.graph import StateGraph
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain.chat_models import init_chat_model

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from confidence_evaluator import calculate_contrast, calculate_confidence, evaluate_relevance, evaluate_universal_metic

load_dotenv()
openai_api_key = os.environ.get("OPENAI_API_KEY")

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
  openai_api_key: str = Field(default="", env="OPENAI_API_KEY")

  # Logging
  log_level: str = "INFO"

  # Computed properties
  @property
  def is_development(self) -> bool:
    return self.environment.lower() == "local"

settings = Settings()
logger = initialize_logger(__name__, settings.is_development, settings.log_level)

async def generate_single_response(messages: list[BaseMessage], model: str, prompt: str, temperature: float, no_stream: bool) -> AIMessage:
  """Single Response Generation -> Confidence Scoring"""

  tags = ["no_stream"] if no_stream else None
  llm = init_chat_model(model=model, api_key=openai_api_key, temperature=temperature, tags=tags)
  custom_messages = messages[:-1] + [HumanMessage(content=prompt)]
  response = await llm.ainvoke(custom_messages)

  return cast("AIMessage", response)

async def generate_responses(state: GraphState) -> GraphState:
  """User Input -> OpenAI Generation -> 3 Responses -> Confidence Scoring"""
  print('generate_responses')

  user_input = state["messages"][-1].content
  # prompt_configs = [
  #   ("openai:gpt-4.1", f"Provide a comprehensive answer: {user_input}", 0.1),
  #   ("openai:gpt-4.1-mini", f"Give a concise, practical response: {user_input}", 0.4),
  #   ("openai:gpt-4.1-nano", f"Offer an innovative perspective: {user_input}", 0.7)
  # ]
  prompt_configs = [
    ("openai:gpt-4.1", user_input, 0.1, False),
    ("openai:gpt-4.1-mini", user_input, 0.4, True),
    ("openai:gpt-4.1-nano", user_input, 0.7, True)
  ]
  # Parallel execution
  tasks = [
    generate_single_response(state["messages"], model, prompt, temp, no_stream)
    for model, prompt, temp, no_stream in prompt_configs
  ]
  responses = await asyncio.gather(*tasks)

  response_contents = [response.content for response in responses]

  return {
    "messages": [responses[0]],
    "responses": response_contents,
    "user_input": user_input
  }

async def generate_feedback_questions(state: GraphState) -> GraphState:
  """Jinja2 Template Processing -> LLM Analysis -> Feedback Generation"""
  print('generate_feedback_questions')
  client = AsyncOpenAI(api_key=settings.openai_api_key or os.environ.get("OPENAI_API_KEY"))
  env = Environment(loader=FileSystemLoader('.'))
  template = env.get_template('feedback-question-generator.md')

  # Template data structure
  template_data = {
    'user_input': state["user_input"],
    'response_1': state["responses"][0] if len(state["responses"]) > 0 else "",
    'response_2': state["responses"][1] if len(state["responses"]) > 1 else "",
    'response_3': state["responses"][2] if len(state["responses"]) > 2 else "",
    'confidence_score': state.get("confidence_score", 0.0),
    'relevance_1': state["relevance_scores"][0] if len(state.get("relevance_scores", [])) > 0 else 0.0,
    'relevance_2': state["relevance_scores"][1] if len(state.get("relevance_scores", [])) > 1 else 0.0,
    'relevance_3': state["relevance_scores"][2] if len(state.get("relevance_scores", [])) > 2 else 0.0,
    'contrast_score': state.get("contrast_score", 0.0),
    'avg_relevance': np.mean(state.get("relevance_scores", [])) if state.get("relevance_scores") else 0.0,
    'peak_relevance': max(state.get("relevance_scores", [])) if state.get("relevance_scores") else 0.0,
  }
  filled_template = template.render(template_data)

  # LLM processing
  response = await client.chat.completions.create(
    model="gpt-4.1",
    messages=[{"role": "user", "content": filled_template}],
    temperature=0.3,
    max_tokens=800
  )
  try:
    feedback_data = json.loads(response.choices[0].message.content)
    feedback_str = "\n\n".join([question["question"] for question in feedback_data["feedback_questions"]])
    return {
      "feedback_questions": feedback_data,
      "messages": [AIMessage(content=feedback_str)]
    }
  except json.JSONDecodeError:
    logger.error("JSON parsing failed")
    return {
      "feedback_questions": {"error": "Generation failed"}
    }
# Graph Construction
def create_response_graph(checkpointer=None):
  """Graph Assembly -> Single Node -> Multi-Response Generator"""
  print('create_response_graph')
  workflow = StateGraph(GraphState)
  # Single processing node
  workflow.add_node("generate_responses", generate_responses)
  workflow.add_node("evaluate_relevance", evaluate_relevance)
  workflow.add_node("calculate_contrast", calculate_contrast)
  workflow.add_node("calculate_confidence", calculate_confidence)
  workflow.add_node("evaluate_universal_metic", evaluate_universal_metic)
  workflow.add_node("generate_feedback_questions", generate_feedback_questions)

  # Graph flow
  workflow.set_entry_point("generate_responses")
  workflow.add_edge("generate_responses", "evaluate_universal_metic")
  workflow.add_edge("generate_responses", "evaluate_relevance")
  workflow.add_edge("generate_responses", "calculate_contrast")
  workflow.add_edge("calculate_contrast", "calculate_confidence")
  workflow.add_edge("evaluate_relevance", "calculate_confidence")
  workflow.add_edge("calculate_confidence", "generate_feedback_questions")
  workflow.add_edge("evaluate_universal_metic", "__end__")

  workflow.set_finish_point("generate_feedback_questions")

  return workflow.compile(checkpointer=checkpointer)

# ---------------
# OUTPUT HELPERS
# ---------------

def display_results(output: Dict):
  """Performance Visualization -> Structured Analytics Display"""
  console = Console()

  # Query header
  query_panel = Panel(
    Text(output["query"], style="bold cyan"),
    title="[bold]Query Analysis[/bold]",
    border_style="blue"
  )
  console.print()
  console.print(query_panel)
  console.print()

  # Response analysis table
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
      content
    )

  console.print(response_table)
  console.print()

  # Performance analytics table
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

def display_feedback_questions(feedback_data: Dict):
  """Feedback Visualization -> Structured Question Display"""
  console = Console()

  # Analysis summary
  summary_panel = Panel(
    Text(f"Intent: {feedback_data.get('analysis_summary', {}).get('primary_intent', 'Unknown')}", style="bold cyan"),
    title="[bold]Analysis Summary[/bold]",
    border_style="blue"
  )
  console.print(summary_panel)
  console.print()

  # Feedback questions table
  questions_table = Table(
    title="Generated Feedback Questions",
    show_header=True,
    header_style="bold magenta"
  )
  questions_table.add_column("Type", style="yellow", width=15)
  questions_table.add_column("Question", style="white", width=60)
  questions_table.add_column("Target", style="green", width=20)

  questions = feedback_data.get("feedback_questions", [])
  for q in questions:
    questions_table.add_row(
      q.get("question_type", "Unknown"),
      q.get("question", "No question"),
      q.get("improvement_target", "No target")
    )
  console.print(questions_table)

# ---------
# MIAN LOOP
# ---------

# Execution framework
async def run_analysis(user_input: str) -> Dict:
  """Input Processing -> Response Generation -> Confidence Assessment"""
  graph = create_response_graph()
  # print(graph.get_graph().draw_mermaid())
  result = await graph.ainvoke({
    "messages": [
      ("human", user_input)
    ],
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
    },
    "feedback_questions": result.get("feedback_questions", {})
  }

async def main():
  output = await run_analysis("Where is the location of 2025 AI Engineer World's Fair Agents Hackathon in SF?")
  display_results(output)
  console = Console()
  console.print("\n" + "=" * 80 + "\n")
  display_feedback_questions(output["feedback_questions"])

if __name__ == "__main__":
  asyncio.run(main())
