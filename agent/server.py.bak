from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
import os
import logging
import json
import asyncio
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
import numpy as np

# LangGraph and core imports
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain.tools import Tool
import subprocess
import yaml

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# In-memory storage for demo
short_term_memory_store = {}
long_term_memory_store = []

# Initialize OpenAI components
openai_api_key = os.environ.get('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)

# Create the main app
app = FastAPI(title="Multi-Agent System with LangGraph & LangMem")
api_router = APIRouter(prefix="/api")

# Agent State Definition
class AgentState(BaseModel):
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
    short_term_memory: List[str] = Field(default_factory=list)
    long_term_memory: List[str] = Field(default_factory=list)
    current_agent: str = "coordinator"
    task_completed: bool = False

# API Models
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class AgentResponse(BaseModel):
    response: str
    agent_used: str
    memory_updated: bool
    session_id: str

# Memory Management Functions
async def store_short_term_memory(session_id: str, content: str):
    """Store content in short-term memory (session-based)"""
    if session_id not in short_term_memory_store:
        short_term_memory_store[session_id] = []
    
    short_term_memory_store[session_id].append({
        "content": content,
        "timestamp": datetime.utcnow(),
        "type": "short_term"
    })
    
    # Keep only last 50 memories per session
    if len(short_term_memory_store[session_id]) > 50:
        short_term_memory_store[session_id] = short_term_memory_store[session_id][-50:]

async def store_long_term_memory(content: str):
    """Store content in long-term memory with vector embeddings"""
    try:
        # Get embedding for the content
        embedding = await embeddings.aembed_query(content)
        
        memory_item = {
            "content": content,
            "embedding": embedding,
            "timestamp": datetime.utcnow(),
            "type": "long_term"
        }
        
        long_term_memory_store.append(memory_item)
        
        # Keep only last 1000 long-term memories
        if len(long_term_memory_store) > 1000:
            long_term_memory_store.pop(0)
        
        return True
    except Exception as e:
        logging.error(f"Error storing long-term memory: {e}")
        return False

async def retrieve_short_term_memory(session_id: str, limit: int = 10):
    """Retrieve recent short-term memories for a session"""
    if session_id not in short_term_memory_store:
        return []
    
    memories = short_term_memory_store[session_id]
    # Return last 'limit' memories
    return [mem["content"] for mem in memories[-limit:]]

async def retrieve_relevant_long_term_memory(query: str, limit: int = 5):
    """Retrieve relevant long-term memories using vector similarity search"""
    try:
        if not long_term_memory_store:
            return []
        
        # Get embedding for the query
        query_embedding = await embeddings.aembed_query(query)
        
        # Calculate cosine similarity for each memory
        similarities = []
        for i, memory in enumerate(long_term_memory_store):
            similarity = cosine_similarity(query_embedding, memory["embedding"])
            similarities.append((similarity, memory["content"]))
        
        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [content for _, content in similarities[:limit]]
        
    except Exception as e:
        logging.error(f"Error retrieving long-term memory: {e}")
        return []

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0

# Tools for agents
def execute_python_code(code: str) -> str:
    """Execute Python code safely and return the result"""
    try:
        # Basic safety check
        if any(dangerous in code.lower() for dangerous in ['import os', 'subprocess', 'eval', 'exec', '__']):
            return "Error: Potentially dangerous code detected"
        
        # Create a safe environment
        safe_globals = {
            '__builtins__': {
                'print': print,
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'list': list,
                'dict': dict,
                'range': range,
                'sum': sum,
                'max': max,
                'min': min,
            }
        }
        
        # Capture output
        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = mystdout = io.StringIO()
        
        try:
            exec(code, safe_globals)
            output = mystdout.getvalue()
        finally:
            sys.stdout = old_stdout
            
        return output if output else "Code executed successfully (no output)"
    except Exception as e:
        return f"Error executing code: {str(e)}"

def parse_json_yaml(content: str) -> str:
    """Parse JSON or YAML content and return structured information"""
    try:
        # Try JSON first
        try:
            data = json.loads(content)
            return f"Valid JSON parsed: {json.dumps(data, indent=2)}"
        except json.JSONDecodeError:
            pass
        
        # Try YAML
        try:
            data = yaml.safe_load(content)
            return f"Valid YAML parsed: {yaml.dump(data, indent=2)}"
        except yaml.YAMLError:
            pass
        
        return "Content is neither valid JSON nor YAML"
    except Exception as e:
        return f"Error parsing content: {str(e)}"

# Agent Tools
python_tool = Tool(
    name="python_executor",
    description="Execute Python code and return the result",
    func=execute_python_code
)

json_yaml_tool = Tool(
    name="json_yaml_parser",
    description="Parse and validate JSON or YAML content",
    func=parse_json_yaml
)

# Agent Functions
async def coordinator_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Main coordinator agent that decides which specialized agent to use"""
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else {}
    user_query = last_message.get("content", "")
    
    # Determine which agent should handle the query
    if "python" in user_query.lower() or "code" in user_query.lower():
        state["current_agent"] = "python_agent"
    elif "json" in user_query.lower() or "yaml" in user_query.lower():
        state["current_agent"] = "data_agent"
    else:
        state["current_agent"] = "general_agent"
    
    # Add coordinator message
    coordinator_msg = {
        "role": "assistant",
        "content": f"Coordinator: Routing to {state['current_agent']} for this task."
    }
    state["messages"].append(coordinator_msg)
    
    return state

async def python_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Agent specialized in Python code execution"""
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else {}
    user_query = last_message.get("content", "")
    
    # Extract Python code from the query
    if "```python" in user_query:
        # Extract code from markdown
        code_start = user_query.find("```python") + 9
        code_end = user_query.find("```", code_start)
        code = user_query[code_start:code_end].strip()
    elif "Execute this Python code:" in user_query:
        # Extract code after the instruction
        code = user_query.split("Execute this Python code:")[-1].strip()
    elif "python code:" in user_query.lower():
        # Extract code after "python code:"
        code = user_query.split("python code:")[-1].strip()
    else:
        # Assume the entire query is code
        code = user_query
    
    result = python_tool.func(code)
    
    response_msg = {
        "role": "assistant",
        "content": f"Python Agent: {result}",
        "agent": "python_agent"
    }
    state["messages"].append(response_msg)
    state["task_completed"] = True
    
    return state

async def data_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Agent specialized in JSON/YAML processing"""
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else {}
    user_query = last_message.get("content", "")
    
    result = json_yaml_tool.func(user_query)
    
    response_msg = {
        "role": "assistant",
        "content": f"Data Agent: {result}",
        "agent": "data_agent"
    }
    state["messages"].append(response_msg)
    state["task_completed"] = True
    
    return state

async def general_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """General purpose agent using LLM"""
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else {}
    user_query = last_message.get("content", "")
    
    # Get relevant memories
    session_id = state.get("context", {}).get("session_id", "default")
    short_term_memories = await retrieve_short_term_memory(session_id)
    long_term_memories = await retrieve_relevant_long_term_memory(user_query)
    
    # Construct context
    context = ""
    if short_term_memories:
        context += "Recent conversation context:\n" + "\n".join(short_term_memories[-3:]) + "\n\n"
    if long_term_memories:
        context += "Relevant knowledge:\n" + "\n".join(long_term_memories) + "\n\n"
    
    # Create LLM prompt
    prompt = f"{context}User query: {user_query}"
    
    try:
        # Use LLM to generate response
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        ai_response = response.content
    except Exception as e:
        ai_response = f"Error: {str(e)}"
    
    response_msg = {
        "role": "assistant",
        "content": f"General Agent: {ai_response}",
        "agent": "general_agent"
    }
    state["messages"].append(response_msg)
    state["task_completed"] = True
    
    return state

# Build the LangGraph workflow
def create_agent_graph():
    """Create the multi-agent workflow graph"""
    workflow = StateGraph(dict)
    
    # Add nodes
    workflow.add_node("coordinator", coordinator_agent)
    workflow.add_node("python_agent", python_agent)
    workflow.add_node("data_agent", data_agent)
    workflow.add_node("general_agent", general_agent)
    
    # Add edges
    workflow.set_entry_point("coordinator")
    
    # Conditional routing based on current_agent
    def route_to_agent(state):
        return state.get("current_agent", "general_agent")
    
    workflow.add_conditional_edges(
        "coordinator",
        route_to_agent,
        {
            "python_agent": "python_agent",
            "data_agent": "data_agent", 
            "general_agent": "general_agent"
        }
    )
    
    # All agents end the workflow
    workflow.add_edge("python_agent", END)
    workflow.add_edge("data_agent", END)
    workflow.add_edge("general_agent", END)
    
    return workflow.compile()

# Initialize the graph
agent_graph = create_agent_graph()

# API Endpoints
@api_router.post("/query", response_model=AgentResponse)
async def process_query(request: QueryRequest):
    """Process a query through the multi-agent system"""
    try:
        session_id = request.session_id or str(uuid.uuid4())
        
        # Store user query in short-term memory
        await store_short_term_memory(session_id, f"User: {request.query}")
        
        # Prepare initial state
        initial_state = {
            "messages": [{"role": "user", "content": request.query}],
            "context": {"session_id": session_id},
            "short_term_memory": [],
            "long_term_memory": [],
            "current_agent": "coordinator",
            "task_completed": False
        }
        
        # Run the agent graph
        result = await agent_graph.ainvoke(initial_state)
        
        # Extract final response
        final_message = result["messages"][-1] if result["messages"] else {}
        response_content = final_message.get("content", "No response generated")
        agent_used = final_message.get("agent", "unknown")
        
        # Store response in short-term memory
        await store_short_term_memory(session_id, f"Assistant: {response_content}")
        
        # Optionally store important information in long-term memory
        if len(request.query) > 50:  # Store longer queries as potentially important
            await store_long_term_memory(f"Query: {request.query}\nResponse: {response_content}")
        
        return AgentResponse(
            response=response_content,
            agent_used=agent_used,
            memory_updated=True,
            session_id=session_id
        )
        
    except Exception as e:
        logging.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@api_router.get("/memory/{session_id}")
async def get_session_memory(session_id: str):
    """Get short-term memory for a session"""
    try:
        memories = await retrieve_short_term_memory(session_id)
        return {"session_id": session_id, "memories": memories}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving memory: {str(e)}")

@api_router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "components": {
            "memory_store": "initialized",
            "agents": "initialized",
            "short_term_sessions": len(short_term_memory_store),
            "long_term_memories": len(long_term_memory_store)
        }
    }

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)