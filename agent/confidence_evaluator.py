from typing import List, cast
from langchain_core.messages import HumanMessage, SystemMessage
import numpy as np
from pydantic import BaseModel
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv
from state import GraphState

load_dotenv()

openai_api_key=os.environ.get("GENAI_OPENAI_API_KEY")


class ContrastEvaluatorOutput(BaseModel):
    contrast_score: float

class RelevanceEvaluatorOutput(BaseModel):
    relevance_scores: List[float]

async def calculate_contrast(state: GraphState) -> GraphState:
    """Calculate similarity scores between all pairs of responses."""

    contrast_prompt = f"""Given the user's question and a list of responses, evaluate contrast between the responses. 
    If the responses are similar, the score should be close to 1. If the responses are different, the score should be close to 0.

    User Question: {state['user_input']}

    Responses to evaluate:
    {"\n".join(f"{i+1}. {resp}" for i, resp in enumerate(state['responses']))}

    For each response, provide a contrast score between 0 and 1.
    """

    messages = [
        SystemMessage(content="You are an expert at evaluating the contrast of responses to questions."),
        HumanMessage(content=contrast_prompt)
    ]
    
    llm = init_chat_model(model="openai:gpt-4.1", api_key=openai_api_key).with_structured_output(ContrastEvaluatorOutput)
    response = cast("ContrastEvaluatorOutput", llm.invoke(messages))
    
    score = response.contrast_score
    
    return {
        'contrast_score': score
    }

async def evaluate_relevance(state: GraphState) -> GraphState:
    """Evaluate how relevant each response is to the user input."""
    print("evaluate_relevance")

    relevance_prompt = f"""Given the user's question and a response, rate how relevant the response is on a scale of 0 to 1.
    A score of 1 means the response perfectly answers the question, while 0 means it's completely irrelevant.

    User Question: {state['user_input']}

    Responses to evaluate:
    {"\n".join(f"{i+1}. {resp}" for i, resp in enumerate(state['responses']))}

    For each response, provide a relevance score between 0 and 1.
    """

    messages = [
        SystemMessage(content="You are an expert at evaluating the relevance of responses to questions."),
        HumanMessage(content=relevance_prompt)
    ]
    
    llm = init_chat_model(model="openai:gpt-4.1", api_key=openai_api_key).with_structured_output(RelevanceEvaluatorOutput)
    response = cast("RelevanceEvaluatorOutput", llm.invoke(messages))
    
    scores = response.relevance_scores
    
    return {
        'relevance_scores': scores
    }


class UniversalMetaEvaluatorOutput(BaseModel):
    metric_key: str 
    eval_prompt: str

class UniversalEvaluatorOutput(BaseModel):
    scores: List[float]
    
async def evaluate_universal_metic(state: GraphState) -> GraphState:
    """Choose an appropriate metric and write LLM-as-judge prompt"""

    system_prompt = """You are: Eval-Prompt Writer, a specialist meta-agent.
Your goal: Given any user_task_prompt (the original task a user posed to an answering agent) you must draft a system prompt for a Judge LLM that will grade the answering agent's response.

Make sure you ask the judge LLM to output a score from 0-1 for the instrucrted metric.
    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"user_task_prompt: {state['user_input']}")
    ]
    
    meta_llm = init_chat_model(model="openai:gpt-4.1", api_key=openai_api_key).with_structured_output(UniversalMetaEvaluatorOutput)
    meta_response = cast("UniversalMetaEvaluatorOutput", meta_llm.invoke(messages))

    llm = init_chat_model(model="openai:gpt-4.1", api_key=openai_api_key).with_structured_output(UniversalEvaluatorOutput)
    content= f"""
user_task_prompt: {state['user_input']}

Responses to evaluate:
{"\n".join(f"{i+1}. {resp}" for i, resp in enumerate(state['responses']))}
"""
    messages = [
        SystemMessage(content=meta_response.eval_prompt),
        HumanMessage(content=content)
    ]
    response = cast("UniversalEvaluatorOutput", llm.invoke(messages))
        
        
    metric_score = np.mean(response.scores) if response.scores else 0.0
    return {
        'universal_eval': {
            "metric_key": meta_response.metric_key,
            "metric_score": float(metric_score),
        }
    }

async def calculate_confidence(state: GraphState) -> GraphState:
    """Calculate final confidence score based on similarity and relevance."""
    print('calculate_confidence')
    contrast_score = state['contrast_score']
    
    # Average relevance score
    avg_relevance = np.mean(state['relevance_scores']) if state['relevance_scores'] else 0.0
    
    # Calculate confidence score
    # We weight both factors equally, but you can adjust the weights
    confidence = (contrast_score + avg_relevance) / 2
    print(f'Confidence: {confidence}')
    state['confidence_score'] = float(confidence)
    return state
