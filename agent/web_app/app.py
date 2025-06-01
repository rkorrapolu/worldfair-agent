import sys
import os
from datetime import datetime
from uuid import uuid4
import asyncio
from typing import AsyncGenerator, Annotated, TypedDict
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from agent import create_response_graph

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

load_dotenv()

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    confidence_score: float


class ChatLangGraph:
    def __init__(self):
        self.graph: CompiledStateGraph = create_response_graph() # type: ignore
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

    async def aquery(self, message: str, thread_id: str | None = None) -> AsyncGenerator[str, None]:
        """
        Async version of query that uses LangGraph's native async streaming.
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

        # Use LangGraph's built-in async streaming to get real-time updates
        try:
            async for chunk in self.graph.astream(input_data, config=config, stream_mode="values"):
                # Check if there are new messages in the state
                if "messages" in chunk and chunk["messages"]:
                    # Get the latest AI message
                    latest_message = chunk["messages"][-1]
                    if hasattr(latest_message, 'content') and latest_message.content:
                        # Yield the content
                        yield latest_message.content
                        break

        except Exception as e:
            yield f"I apologize, but I encountered an error: {str(e)}"

        # Update thread metadata
        self.active_threads[thread_id]["message_count"] += 2  # Human + AI message

    async def astream_confidence(self, thread_id: str) -> AsyncGenerator[dict, None]:
        """
        Stream confidence updates for a specific thread using LangGraph's native streaming.
        This creates a passive listener that streams confidence updates only during active chat.
        """
        if thread_id not in self.active_threads:
            raise KeyError(f"Thread {thread_id} not found")

        config = RunnableConfig(configurable={"thread_id": thread_id})

        try:
            # Send initial confidence value
            current_confidence = self.get_thread_confidence(thread_id)
            yield {
                "type": "confidence_update",
                "thread_id": thread_id,
                "confidence": current_confidence,
            }

            # Stream confidence updates from LangGraph's native streaming
            # This will only emit updates when the graph is actively running
            async for chunk in self.graph.astream(
                {},  # Empty input to just listen for state changes
                config=config,
                stream_mode="values"
            ):
                if "confidence_score" in chunk:
                    yield {
                        "type": "confidence_update",
                        "thread_id": thread_id,
                        "confidence": chunk["confidence_score"],
                    }

        except Exception as e:
            yield {
                "type": "error",
                "message": str(e),
                "thread_id": thread_id,
            }


# Remove the broadcast_confidence_update function since LangGraph handles streaming natively

# Initialize the chatbot
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
            async for token in bot.aquery(chat_message.message, chat_message.thread_id):
                full_response += token
                # Send each token as a streaming event
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                # Small delay to prevent overwhelming the client
                await asyncio.sleep(0.01)

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


@app.get("/api/confidence/stream")
async def confidence_stream(
    thread_id: str, bot: Annotated[ChatLangGraph, Depends(get_chatbot)]
):
    """Stream real-time confidence updates using LangGraph's native streaming capabilities."""

    async def generate_confidence_stream() -> AsyncGenerator[str, None]:
        try:
            async for confidence_data in bot.astream_confidence(thread_id):
                yield f"data: {json.dumps(confidence_data)}\n\n"

        except KeyError:
            # Thread doesn't exist
            error_data = {"type": "error", "message": "Thread not found", "thread_id": thread_id}
            yield f"data: {json.dumps(error_data)}\n\n"
        except Exception as e:
            # Send error event
            error_data = {"type": "error", "message": str(e), "thread_id": thread_id}
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(
        generate_confidence_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        },
    )
