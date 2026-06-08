# Chapter 6: The Filesystem Middleware — How Agents Remember

> *"The filesystem middleware is what separates a goldfish agent from an elephant agent. One forgets everything after each tool call. The other writes it down."*

---

## What You'll Learn

- The complete filesystem API: `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`
- How the virtual filesystem maps to agent state
- Short-term vs long-term memory in DeepAgents
- All four backends: State, Store, Filesystem, Composite
- How to configure cross-session agent memory
- Advanced: agent "memoirs" — agents that learn from past sessions

---

## 6.1 The Filesystem Tool Reference

Every deep agent gets these filesystem tools automatically:[cite:57]

```python
# The 6 filesystem tools explained with examples

# 1. ls - List directory contents
# Agent uses this to navigate the virtual filesystem
ls("/")              # List root directory
ls("/research/")     # List a subdirectory

# Returns:
# ["/research/notes.txt", "/research/data.json", "/drafts/"]

# 2. read_file - Read a file
read_file("/research/notes.txt")
# Returns: full file contents as string

read_file("/large_report.txt", start_line=1, end_line=50)
# Returns: only lines 1-50 (useful for large files!)

# 3. write_file - Create or overwrite a file
write_file(
    path="/research/notes.txt",
    content="Key findings:\n1. AI market is $500B\n2. ..."
)
# Returns: confirmation message

# 4. edit_file - Edit an existing file without rewriting it
edit_file(
    path="/research/notes.txt",
    old_str="1. AI market is $500B",
    new_str="1. AI market is $638B (2025 estimate)"
)
# Returns: confirmation or 'string not found'

# 5. glob - Find files matching a pattern
glob("/research/**/*.txt")    # All .txt files in /research/ recursively
glob("/drafts/chapter_*.md") # All chapter markdown files
# Returns: list of matching paths

# 6. grep - Search file contents
grep("/", "Section 80C", include="*.txt")  # Find 'Section 80C' in all .txt files
# Returns: matching lines with file paths and line numbers
```

---

## 6.2 The Virtual Filesystem Mental Model

The "virtual filesystem" in DeepAgents is not your real computer's filesystem (by default). It's a dictionary stored in the agent's LangGraph state:

```python
# Internally, the StateBackend stores files like this:
filesystem_state = {
    "/notes.txt": {"content": "My notes...", "encoding": "utf-8"},
    "/research/ai_trends.txt": {"content": "AI trends...", "encoding": "utf-8"},
    "/data/numbers.json": {"content": '{"value": 42}', "encoding": "utf-8"},
}

# The agent calls write_file("/notes.txt", "My notes...")
# DeepAgents updates filesystem_state["/notes.txt"]
# The agent calls read_file("/notes.txt")
# DeepAgents reads from filesystem_state["/notes.txt"]

# This state travels with the agent's conversation history
# in the LangGraph state graph
```

This is why files "disappear" between separate runs without a checkpointer — they only live in the current thread's state.

---

## 6.3 Backend Deep Dive

### Backend 1: StateBackend (Default)

```python
from deepagents import create_deep_agent

# Default setup - uses StateBackend automatically
agent = create_deep_agent(model="openai:gpt-4o")

# Files live in agent state - ephemeral per thread
# Pros: Zero setup, fast, no external dependencies
# Cons: Files disappear when thread ends
```

### Backend 2: StoreBackend (Cross-Session Persistence)

```python
from deepagents import create_deep_agent
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.backends import StoreBackend
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver

store = InMemoryStore()        # In production: SQLiteStore or RedisStore
checkpointer = MemorySaver()   # In production: SqliteSaver or RedisSaver

agent = create_deep_agent(
    model="openai:gpt-4o",
    store=store,
    checkpointer=checkpointer,
    middleware=[
        FilesystemMiddleware(
            backend=lambda rt: StoreBackend(rt)
        )
    ]
)

# Now files persist across different thread_ids!
config_a = {"configurable": {"thread_id": "session-A"}}
config_b = {"configurable": {"thread_id": "session-B"}}

# Session A writes a file
agent.invoke({"messages": "Write my name (Yash) to /user/profile.txt"}, config=config_a)

# Session B can read it!
result = agent.invoke({"messages": "Read /user/profile.txt and tell me what's there"}, config=config_b)
print(result["messages"][-1].content)  # "The file contains 'Yash'"
```

### Backend 3: FilesystemBackend (Real Disk Access)

```python
from deepagents.backends import FilesystemBackend
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents import create_deep_agent
import os

# Agent reads/writes REAL files on your computer
agent = create_deep_agent(
    model="openai:gpt-4o",
    middleware=[
        FilesystemMiddleware(
            backend=lambda rt: FilesystemBackend(
                root_dir="/home/yash/agent_workspace/",  # Agent is sandboxed here
            )
        )
    ]
)

# Now when the agent calls write_file("/report.txt", ...) it writes to:
# /home/yash/agent_workspace/report.txt
# on your REAL filesystem!

result = agent.invoke({
    "messages": "Write a 3-point summary of GST input tax credit rules to /gst_notes.txt"
})
print(result["messages"][-1].content)
# Check your filesystem: /home/yash/agent_workspace/gst_notes.txt now exists!
```

### Backend 4: CompositeBackend (Best of Both)

```python
# chapter_06/composite_backend.py
# The recommended production setup

from deepagents import create_deep_agent
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver

store = InMemoryStore()
checkpointer = MemorySaver()

def make_backend(runtime):
    """Route different paths to different backends."""
    return CompositeBackend(
        default=StateBackend(runtime),      # Most files: fast, ephemeral
        routes={
            "/memories/": StoreBackend(runtime),  # Long-term memory: persistent
            "/session/": StateBackend(runtime),   # Session data: ephemeral
        }
    )

agent = create_deep_agent(
    model="openai:gpt-4o",
    store=store,
    checkpointer=checkpointer,
    middleware=[
        FilesystemMiddleware(
            backend=make_backend,
            custom_tool_descriptions={
                "write_file": (
                    "Write to files. "
                    "Use /memories/ prefix for information that should persist across sessions. "
                    "Use /session/ for temporary working data."
                ),
            }
        )
    ],
    system_prompt="""
For memory management:
- Save important facts about the user to /memories/user_profile.txt
- Save research results to /memories/research/<topic>.txt  
- Use /session/ for temporary calculations and drafts
- At the start of each conversation, check /memories/ to recall user context
"""
)
```

---

## 6.4 Building an Agent with Persistent Memory

```python
# chapter_06/persistent_memory_agent.py
# A complete example of a persistent memory agent

from dotenv import load_dotenv
load_dotenv()

from deepagents import create_deep_agent
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver

# Setup persistent storage
shared_store = InMemoryStore()  # Replace with SQLite/Redis in production
checkpointer = MemorySaver()

# Create a personal assistant that remembers you
assistant = create_deep_agent(
    model="openai:gpt-4o",
    store=shared_store,
    checkpointer=checkpointer,
    middleware=[
        FilesystemMiddleware(
            backend=lambda rt: CompositeBackend(
                default=StateBackend(rt),
                routes={"/memories/": StoreBackend(rt)}
            )
        )
    ],
    system_prompt="""
You are a personal assistant with persistent memory.

At the start of EVERY conversation:
1. Call ls("/memories/") to see what you remember
2. If /memories/user_profile.txt exists, read it to recall who the user is
3. Use this context to personalize your responses

During conversations:
- When the user shares personal information (name, job, preferences, goals), 
  update /memories/user_profile.txt
- When you complete research, save key findings to /memories/research/<topic>.txt
- When the user mentions recurring tasks, save them to /memories/tasks.txt

Format /memories/user_profile.txt as:
  Name: <name>
  Profession: <job>
  Location: <city, country>
  Preferences: <key preferences>
  Goals: <main goals>
"""
)

# First conversation
print("=== Session 1 ===")
result1 = assistant.invoke(
    {"messages": "Hi! I'm Yash, a tax lawyer in Indore. I specialize in ITAT cases."},
    config={"configurable": {"thread_id": "yash-session-1"}}
)
print(result1["messages"][-1].content)

# Different session - new thread ID but same store
print("\n=== Session 2 (new session) ===")
result2 = assistant.invoke(
    {"messages": "Hello! Who am I again?"},
    config={"configurable": {"thread_id": "yash-session-2"}}  # Different thread!
)
print(result2["messages"][-1].content)
# Should recall: "You're Yash, a tax lawyer in Indore specializing in ITAT cases"
```

---

## Summary

- The filesystem middleware gives agents 6 tools: `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`
- By default, files are ephemeral (StateBackend) — they live only in the current thread
- For cross-session memory, use StoreBackend with a persistent store
- CompositeBackend routes different paths to different backends (recommended for production)
- The system prompt instructs the agent WHEN and HOW to use the filesystem

---

[← Chapter 5](../chapter-05/chapter-05.md) | [Chapter 7 →](../chapter-07/chapter-07.md)
