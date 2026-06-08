# Chapter 9: Running Deep Agents Locally with Ollama

> *"Privacy matters. Cost matters. Speed matters. Running AI locally solves all three. In 2025, local models are good enough for most real agent tasks."*

---

## What You'll Learn

- Which local models work best with DeepAgents
- How to use Ollama as the agent's LLM backend
- Performance comparison: local vs cloud
- Workarounds for local model limitations
- The DeepAgents CLI with Ollama models
- Privacy-first agent design

---

## 9.1 Why Run Locally?

| Consideration | Cloud API | Local (Ollama) |
|--------------|-----------|----------------|
| **Cost** | Pay per token | Free forever |
| **Privacy** | Data leaves your device | Never leaves your machine |
| **Speed** | Fast (ms response) | Depends on hardware |
| **Quality** | Best (GPT-4o, Claude) | Good-Great (Qwen, Llama) |
| **Offline** | Requires internet | Works offline |
| **Limits** | Rate limits/quotas | No limits |

For a tax lawyer in India handling confidential client data, **local models are often the right choice** for sensitive documents.

---

## 9.2 Setting Up Ollama for DeepAgents

```bash
# Pull the best models for agent use

# Option 1: Qwen2.5:14b - Best quality/performance ratio for agents
ollama pull qwen2.5:14b

# Option 2: Qwen2.5:7b - Faster, needs less RAM
ollama pull qwen2.5:7b  

# Option 3: Llama3.1:8b - Good general purpose
ollama pull llama3.1:8b

# Option 4: Mistral:7b - Excellent for instruction following
ollama pull mistral:7b

# Check what you have
ollama list
```

---

## 9.3 Using Ollama with DeepAgents

```python
# chapter_09/ollama_agent.py

from dotenv import load_dotenv
load_dotenv()

from deepagents import create_deep_agent
from langchain.tools import tool

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression. Use for any arithmetic."""
    try:
        result = eval(expression, {"__builtins__": {}, "abs": abs, "round": round})
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error: {e}"

# Replace 'openai:gpt-4o' with 'ollama:model_name'
agent = create_deep_agent(
    model="ollama:qwen2.5:14b",  # Using local Ollama model!
    tools=[calculate],
    system_prompt="You are a helpful assistant. Use tools when appropriate."
)

result = agent.invoke({
    "messages": "Calculate: What is 15% GST on a base amount of Rs. 85,000?"
})
print(result["messages"][-1].content)
```

---

## 9.4 Ollama with Custom Model Names (Workaround)

If you want to use Ollama but the DeepAgents CLI expects OpenAI/Anthropic model prefixes, here's the workaround used by the community:[cite:38]

```bash
# Create a custom Modelfile that maps an Ollama model to a 'gpt-' prefixed name
# This tricks tools that only accept 'gpt-' or 'claude-' model names

# Create Modelfile
cat > Modelfile << 'EOF'
FROM qwen2.5:14b
# This creates a model accessible as 'gpt-5-mini' in Ollama
# But it's actually qwen2.5:14b running locally!
EOF

# Build the aliased model
ollama create gpt-5-mini -f Modelfile

# Now run with the alias
ollama run gpt-5-mini "Test prompt"
```

```python
# Use environment variables to redirect OpenAI calls to Ollama
import os
os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"
os.environ["OPENAI_BASE_URL"] = "http://localhost:11434/v1"
os.environ["OPENAI_API_KEY"] = "ollama"  # Not actually used but required

# Now code written for OpenAI will use Ollama instead!
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-5-mini")  # Actually runs Qwen2.5:14b locally!
```

---

## 9.5 Local Model Performance Guide

Not all local models handle the agent loop equally well. Here's what I've tested:

| Model | Tool Calling | Planning | Context | Rec. For |
|-------|-------------|----------|---------|----------|
| qwen2.5:14b | ✅ Excellent | ✅ Good | 32K | Best local agent model |
| qwen2.5:7b | ✅ Good | ✅ OK | 32K | Lighter tasks |
| llama3.1:8b | ✅ Good | ✅ Good | 128K | Long context needs |
| mistral:7b | ⚠️ Mixed | ⚠️ Basic | 32K | Simple tasks only |
| phi3:mini | ❌ Poor | ❌ Poor | 4K | Not recommended for agents |

**Recommendation:** Use `qwen2.5:14b` for agent tasks if your hardware supports it.

---

## 9.6 A Complete Local Privacy-First Agent

```python
# chapter_09/private_local_agent.py
# A fully local, privacy-first agent for sensitive documents

from deepagents import create_deep_agent
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.backends import FilesystemBackend
from langchain.tools import tool
import os

# All files stored on LOCAL disk - nothing goes to cloud!
WORKSPACE = os.path.expanduser("~/private_agent_workspace")
os.makedirs(WORKSPACE, exist_ok=True)

@tool
def read_local_document(filename: str) -> str:
    """Read a document from the local workspace.
    Use this to read uploaded client documents, tax files, etc.
    Args:
        filename: Name of the file in the workspace (e.g., 'assessment_order.txt')
    """
    filepath = os.path.join(WORKSPACE, filename)
    if not os.path.exists(filepath):
        return f"File not found: {filename}. Available files: {os.listdir(WORKSPACE)}"
    with open(filepath, 'r') as f:
        return f.read()

@tool  
def list_documents() -> str:
    """List all documents in the workspace."""
    files = os.listdir(WORKSPACE)
    return f"Documents in workspace: {files}" if files else "Workspace is empty."

# All computation is LOCAL - Ollama + local filesystem
private_agent = create_deep_agent(
    model="ollama:qwen2.5:14b",  # ✅ Local model
    tools=[read_local_document, list_documents],
    middleware=[
        FilesystemMiddleware(
            backend=lambda rt: FilesystemBackend(
                root_dir=WORKSPACE    # ✅ Local filesystem
            )
        )
    ],
    system_prompt="""
You are a confidential legal and tax document assistant.

Privacy Protocol:
- ALL processing happens locally - never send data to external services
- Use read_local_document to access client files
- Save analysis to /analysis/ folder within the workspace
- When analyzing tax documents, identify: taxpayer, AY, section, amounts
- For assessment orders: identify additions made and grounds

You specialize in Income Tax Act analysis for Indian tax practitioners.
"""
)

# Test with a sample analysis
result = private_agent.invoke({
    "messages": """
    List available documents and provide a framework for analyzing 
    an income tax assessment order under Section 143(3).
    """
})
print(result["messages"][-1].content)
```

---

## 9.7 Handling Local Model Limitations

Local models sometimes struggle with complex tool call formatting. Here are fixes:

```python
# chapter_09/local_model_fixes.py

from deepagents import create_deep_agent

# Tip 1: Use simpler tool signatures for local models
# Instead of complex nested types, use simple str/int/float

# BAD for local models:
from typing import Optional, List, Dict
# def complex_search(params: Dict[str, List[Optional[str]]]) -> dict:

# GOOD for local models:
from langchain.tools import tool

@tool
def simple_search(query: str) -> str:
    """Search for information. query: the search term."""
    return f"Results for: {query}"  # Simplified

# Tip 2: Add explicit tool usage examples to system_prompt
agent = create_deep_agent(
    model="ollama:qwen2.5:14b",
    tools=[simple_search],
    system_prompt="""
When you need information, use the simple_search tool.
Example usage: search for 'GST rate jewelry' using the simple_search tool.
Always call tools explicitly rather than answering from memory.
"""
)

# Tip 3: Lower temperature for more consistent tool calling
from langchain_ollama import ChatOllama
from deepagents import create_deep_agent

local_llm = ChatOllama(
    model="qwen2.5:14b",
    temperature=0.1,  # Lower temperature = more consistent tool calling
)

agent = create_deep_agent(
    model=local_llm,  # Can pass a model object directly instead of string!
    tools=[simple_search],
    system_prompt="Use tools explicitly to answer questions."
)
```

---

## Summary

- Ollama enables free, private, offline AI agents
- `qwen2.5:14b` is the best local model for agent tasks
- Use `model="ollama:model_name"` in `create_deep_agent()`
- For privacy-sensitive work (legal documents, client data), local models are ideal
- Workaround for model name restrictions: create Ollama aliases
- Keep tool signatures simple for best local model compatibility

---

[← Chapter 8](../chapter-08/chapter-08.md) | [Chapter 10 →](../chapter-10/chapter-10.md)
