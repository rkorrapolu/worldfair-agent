from typing import List, cast
from langchain_core.messages import HumanMessage, SystemMessage
import numpy as np
from pydantic import BaseModel
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv
from state import GraphState

load_dotenv()

openai_api_key=os.environ.get("OPENAI_API_KEY")


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
    
    state['contrast_score'] = score
    return state

async def evaluate_relevance(state: GraphState) -> GraphState:
    """Evaluate how relevant each response is to the user input."""

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
    
    state['relevance_scores'] = scores
    return state

async def calculate_confidence(state: GraphState) -> GraphState:
    """Calculate final confidence score based on similarity and relevance."""
    contrast_score = state['contrast_score']
    
    # Average relevance score
    avg_relevance = np.mean(state['relevance_scores']) if state['relevance_scores'] else 0.0
    
    # Calculate confidence score
    # We weight both factors equally, but you can adjust the weights
    confidence = (contrast_score + avg_relevance) / 2
    
    state['confidence_score'] = float(confidence)
    return state
