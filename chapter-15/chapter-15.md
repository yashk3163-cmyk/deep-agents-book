# Chapter 15: Deploying Your Agent to Production

> *"Building an agent that works on your laptop is the beginning. Making it reliable, observable, and scalable for real users — that's the craft."*

---

## What You'll Learn

- Wrapping your agent in a FastAPI web server
- Adding LangSmith for observability and tracing
- Handling errors gracefully
- Rate limiting and security
- Dockerizing your agent
- Deployment options (Cloud Run, Railway, Fly.io)

---

## 15.1 Wrapping Your Agent in FastAPI

```python
# chapter_15/agent_server.py
# Turn any DeepAgent into a REST API server

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
import json

from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver

# --- Setup ---
app = FastAPI(
    title="DeepAgent API",
    description="Your custom AI agent served as a REST API",
    version="1.0.0"
)

checkpointer = MemorySaver()

agent = create_deep_agent(
    model="openai:gpt-4o",
    checkpointer=checkpointer,
    system_prompt="You are a helpful AI assistant."
)

# --- Request/Response Models ---
class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = "default"
    stream: Optional[bool] = False

class ChatResponse(BaseModel):
    response: str
    thread_id: str
    message_count: int

# --- Endpoints ---

@app.get("/health")
async def health_check():
    """Health check endpoint for deployment platforms."""
    return {"status": "healthy", "agent": "running"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message to the agent and get a response."""
    try:
        config = {"configurable": {"thread_id": request.thread_id}}
        
        result = agent.invoke(
            {"messages": request.message},
            config=config
        )
        
        response_text = result["messages"][-1].content
        
        return ChatResponse(
            response=response_text,
            thread_id=request.thread_id,
            message_count=len(result["messages"])
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream the agent's response token by token."""
    config = {"configurable": {"thread_id": request.thread_id}}
    
    def generate():
        for chunk in agent.stream(
            {"messages": request.message},
            config=config,
            stream_mode="updates"
        ):
            if "agent" in chunk:
                messages = chunk["agent"].get("messages", [])
                for msg in messages:
                    if hasattr(msg, 'content') and msg.content and msg.type == 'ai':
                        data = {"content": msg.content, "done": False}
                        yield f"data: {json.dumps(data)}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

# --- Run ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

```bash
# Install and run
pip install fastapi uvicorn
python chapter_15/agent_server.py

# Test:
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!", "thread_id": "user123"}'
```

---

## 15.2 Adding LangSmith Observability

LangSmith lets you see every step your agent takes, debug failures, and evaluate performance:[cite:9]

```python
# Add to your .env file:
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=your_langsmith_key
# LANGCHAIN_PROJECT=my-agent-project

# That's it! With those env variables set, every agent run is automatically
# traced to LangSmith. You'll see:
# - Every tool call and its result
# - Token usage per step
# - Latency per step
# - Full conversation history
# - Any errors or exceptions

# No code changes required - LangChain/DeepAgents auto-detect these variables
```

---

## 15.3 Dockerfile for Deployment

```dockerfile
# chapter_15/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Don't run as root in production
RUN adduser --disabled-password --gecos '' appuser
USER appuser

# Expose port
EXPOSE 8000

# Start the server
CMD ["python", "chapter_15/agent_server.py"]
```

```bash
# Build and run
docker build -t my-agent .
docker run -p 8000:8000 --env-file .env my-agent
```

---

## 15.4 Deployment Options

| Platform | Cost | Ease | Best For |
|----------|------|------|----------|
| **Railway** | $5/mo | Very Easy | Side projects |
| **Fly.io** | Free tier | Easy | Light production |
| **Google Cloud Run** | Pay per use | Medium | Scalable production |
| **AWS Lambda** | Pay per use | Hard | High volume |
| **Render** | Free tier | Easy | Hobby projects |

**Quickest deployment: Railway**
```bash
npm install -g @railway/cli
railway login
railway up
```

---

## Summary

You now know how to:
- Serve any agent as a REST API with FastAPI
- Stream responses for responsive UIs
- Add full observability with LangSmith
- Containerize with Docker
- Deploy to cloud platforms

**Congratulations** — you've completed the book. You can now build, customize, and deploy production AI agents.

---

[← Chapter 14](../chapter-14/chapter-14.md) | [Appendix A →](../appendix-a/appendix-a.md)
