import sys
import os
from datetime import datetime
from uuid import uuid4
import asyncio
from typing import AsyncGenerator, Annotated, TypedDict
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from agent import create_response_graph, create_simple_chat_graph

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import json

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    confidence_score: float


class ChatLangGraph:
    def __init__(self):
        self.checkpointer = MemorySaver()
        # Import the simple chat graph for streaming
        self.graph: CompiledStateGraph = create_simple_chat_graph(checkpointer=self.checkpointer) # type: ignore
        self.active_threads = {}

    def create_new_thread(self) -> str:
        """Create a new conversation thread."""
        thread_id = str(uuid4())
        self.active_threads[thread_id] = {
            "title": f"Chat {len(self.active_threads) + 1}",
            "created_at": datetime.now(),
            "message_count": 0,
            "confidence_score": 1.0,
        }
        return thread_id

    def list_threads(self) -> dict:
        """List all active threads."""
        return self.active_threads

    def get_thread_message_count(self, thread_id: str) -> int:
        """Get message count for a thread."""
        if thread_id not in self.active_threads:
            return 0
        return self.active_threads[thread_id]["message_count"]

    def get_thread_confidence(self, thread_id: str) -> float:
        """Get confidence score for a thread."""
        if thread_id not in self.active_threads:
            raise KeyError(f"Thread {thread_id} not found")

        # Get the latest state from the graph
        config = RunnableConfig(configurable={"thread_id": thread_id})
        state = self.graph.get_state(config)
        return state.values.get("confidence_score", 1.0)

    def get_thread_messages(self, thread_id: str) -> list:
        """Get conversation history for a thread."""
        if thread_id not in self.active_threads:
            raise KeyError(f"Thread {thread_id} not found")

        # Get message history from the graph state
        config = RunnableConfig(configurable={"thread_id": thread_id})
        state = self.graph.get_state(config)
        if state.values and "messages" in state.values:
            return state.values["messages"]
        return []

    def delete_thread(self, thread_id: str) -> None:
        """Delete a conversation thread."""
        if thread_id not in self.active_threads:
            raise KeyError(f"Thread {thread_id} not found")
        del self.active_threads[thread_id]

    async def aquery(self, message: str, thread_id: str | None = None) -> AsyncGenerator[dict, None]:
        """
        Async version of query that uses LangGraph's native async streaming.
        Yields both chat content and confidence updates.
        """
        # Ensure thread exists
        if thread_id and thread_id not in self.active_threads:
            raise KeyError(f"Thread {thread_id} not found")

        if not thread_id:
            thread_id = self.create_new_thread()

        # Use LangGraph implementation with async streaming
        input_data = {
            "messages": [HumanMessage(content=message)],
            "confidence_score": 1.0,
        }

        config = RunnableConfig(configurable={"thread_id": thread_id})

        # Use messages mode to stream LLM tokens directly
        async for chunk in self.graph.astream(input_data, config=config, stream_mode="messages"):
            message_chunk, metadata = chunk
            # Stream individual tokens from the LLM
            if hasattr(message_chunk, 'content') and message_chunk.content:
                yield {
                    "type": "token",
                    "content": message_chunk.content,
                }

        # Also get the final state for confidence updates
        async for chunk in self.graph.astream(input_data, config=config, stream_mode="values"):
            if "confidence_score" in chunk:
                yield {
                    "type": "confidence_update",
                    "thread_id": thread_id,
                    "confidence": chunk["confidence_score"],
                }
                break

        self.active_threads[thread_id]["message_count"] += 2  # Human + AI message


chatbot: ChatLangGraph | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global chatbot
    chatbot = ChatLangGraph()
    yield
    # Shutdown (cleanup if needed)
    chatbot = None


app = FastAPI(
    lifespan=lifespan,
)

web_app_dir = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(web_app_dir, "templates"))
app.mount(
    "/static", StaticFiles(directory=os.path.join(web_app_dir, "static")), name="static"
)


# Dependency to get chatbot instance
def get_chatbot() -> ChatLangGraph:
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    return chatbot


class ChatMessage(BaseModel):
    message: str
    thread_id: str | None = None


class ThreadCreate(BaseModel):
    title: str | None = None


class ThreadResponse(BaseModel):
    thread_id: str
    title: str


class ThreadInfo(BaseModel):
    thread_id: str
    title: str
    message_count: int
    created_at: str


class ThreadsResponse(BaseModel):
    threads: list[ThreadInfo]


class MessageInfo(BaseModel):
    type: str
    content: str
    timestamp: str | None = None


class ThreadHistoryResponse(BaseModel):
    messages: list[MessageInfo]


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/threads", response_model=ThreadResponse, status_code=201)
async def create_thread(
    thread_data: ThreadCreate, bot: Annotated[ChatLangGraph, Depends(get_chatbot)]
):
    """Create a new conversation thread."""
    thread_id = bot.create_new_thread()
    if thread_data.title:
        bot.active_threads[thread_id]["title"] = thread_data.title
    return ThreadResponse(
        thread_id=thread_id, title=bot.active_threads[thread_id]["title"]
    )


@app.get("/api/threads", response_model=ThreadsResponse)
async def list_threads(bot: Annotated[ChatLangGraph, Depends(get_chatbot)]):
    """List all conversation threads."""
    threads = []
    for thread_id, info in bot.list_threads().items():
        threads.append(
            ThreadInfo(
                thread_id=thread_id,
                title=info["title"],
                message_count=bot.get_thread_message_count(thread_id),
                created_at=info["created_at"].isoformat(),
            )
        )
    return ThreadsResponse(threads=threads)


@app.get("/api/threads/{thread_id}/history", response_model=ThreadHistoryResponse)
async def get_thread_history(
    thread_id: str, bot: Annotated[ChatLangGraph, Depends(get_chatbot)]
):
    """Get conversation history for a specific thread."""
    history = bot.get_thread_messages(thread_id)
    messages = []
    for msg in history:
        messages.append(
            MessageInfo(
                type=msg.type,
                content=msg.content,
                timestamp=getattr(msg, 'timestamp', None),
            )
        )
    return ThreadHistoryResponse(messages=messages)


@app.delete("/api/threads/{thread_id}", status_code=204)
async def delete_thread(
    thread_id: str, bot: Annotated[ChatLangGraph, Depends(get_chatbot)]
):
    """Delete a conversation thread."""
    try:
        bot.delete_thread(thread_id)
        return None  # 204 No Content
    except KeyError:
        raise HTTPException(status_code=404, detail="Thread not found")


@app.post("/api/chat/stream")
async def chat_stream(
    chat_message: ChatMessage, bot: Annotated[ChatLangGraph, Depends(get_chatbot)]
):
    """Stream chat response using Server-Sent Events."""

    async def generate_response() -> AsyncGenerator[str, None]:
        try:
            # Send initial event to indicate streaming started
            yield f"data: {json.dumps({'type': 'start'})}\n\n"

            # Collect the full response for potential use
            full_response = ""

            # Stream the response from the chatbot
            async for update in bot.aquery(chat_message.message, chat_message.thread_id):
                if update["type"] == "token":
                    # Handle individual LLM tokens - no need to break them down further
                    content = update["content"]
                    full_response += content
                    # Send the token directly as received from LLM
                    yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"
                    
                elif update["type"] == "confidence_update":
                    # Handle confidence updates
                    yield f"data: {json.dumps(update)}\n\n"

            # Send completion event
            yield f"data: {json.dumps({'type': 'complete', 'full_response': full_response})}\n\n"

        except Exception as e:
            # Send error event
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
