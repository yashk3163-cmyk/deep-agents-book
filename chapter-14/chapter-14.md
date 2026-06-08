# Chapter 14: Memory, Persistence and Long-Running Agents

> *"An agent that forgets is just a very expensive autocomplete. The real value of AI agents is accumulation: knowledge that compounds over time."*

---

## What You'll Learn

- The three types of agent memory
- Checkpointing: saving and resuming agent state
- Cross-session persistence with LangGraph Store
- SQLite persistence for production-like setups
- Agent memory architecture patterns
- How to make an agent that gets smarter over time

---

## 14.1 The Three Types of Agent Memory

```
Type 1: In-Context Memory (Conversation History)
  - What: The list of messages in the current conversation
  - Scope: Current thread only, until context window limit
  - Backend: Always available, no setup needed
  - Analogy: Working memory / RAM

Type 2: In-State Memory (Checkpointed State)
  - What: Agent's filesystem + conversation, saved between calls  
  - Scope: Per thread_id, survives multiple invoke() calls
  - Backend: MemorySaver (RAM) or SqliteSaver (disk)
  - Analogy: Current project folder / short-term storage

Type 3: Cross-Session Memory (Store/Database)
  - What: Persistent knowledge store, searchable
  - Scope: Across ALL threads and sessions
  - Backend: InMemoryStore (RAM), SqliteStore, Redis, PostgreSQL
  - Analogy: Long-term knowledge base / permanent memory
```

---

## 14.2 Checkpointing: Resuming Long Tasks

```python
# chapter_14/checkpointing_demo.py

from dotenv import load_dotenv
load_dotenv()

from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver
# For real persistence: from langgraph.checkpoint.sqlite import SqliteSaver

checkpointer = MemorySaver()
# Real persistence: checkpointer = SqliteSaver.from_conn_string("agent_state.db")

agent = create_deep_agent(
    model="openai:gpt-4o",
    checkpointer=checkpointer,
    system_prompt="You are a research assistant. Remember all context."
)

config = {"configurable": {"thread_id": "long-research-task"}}

# First invocation - start the task
result1 = agent.invoke(
    {"messages": "Start researching AI trends. Focus on agents first."},
    config=config
)
print("Step 1 done:", result1["messages"][-1].content[:100])

# Second invocation - SAME thread, agent continues where it left off
result2 = agent.invoke(
    {"messages": "Good. Now research AI infrastructure trends."},
    config=config
)
print("Step 2 done:", result2["messages"][-1].content[:100])

# Third - agent has full context of steps 1 and 2
result3 = agent.invoke(
    {"messages": "Now synthesize everything into a final report."},
    config=config
)
print("Final report:", result3["messages"][-1].content[:200])
```

---

## 14.3 Cross-Session Long-Term Memory

```python
# chapter_14/long_term_memory.py
# An agent that builds knowledge over many sessions

from dotenv import load_dotenv
load_dotenv()

from deepagents import create_deep_agent
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver

# These are the two key objects for long-term memory
store = InMemoryStore()      # For cross-session knowledge (replace with SQLite for persistence)
checkpointer = MemorySaver() # For per-session state

knowledge_agent = create_deep_agent(
    model="openai:gpt-4o",
    store=store,
    checkpointer=checkpointer,
    middleware=[
        FilesystemMiddleware(
            backend=lambda rt: CompositeBackend(
                default=StateBackend(rt),
                routes={
                    "/knowledge/": StoreBackend(rt),   # Persists across all sessions
                    "/session/": StateBackend(rt),      # Current session only
                }
            )
        )
    ],
    system_prompt="""
You are a knowledge accumulation agent. Each session you learn new things.

MEMORY PROTOCOL:
1. Session start: ls('/knowledge/') to see what you know
2. During session: When you learn something important, SAVE it:
   - Topic summaries: /knowledge/<topic>.md
   - Key facts: /knowledge/facts.md (append new facts)
   - User context: /knowledge/user/<user_id>.md
3. Always date your entries: use the current date in filenames or headers

KNOWLEDGE ORGANIZATION:
/knowledge/
  topics/          <- One file per topic
  facts.md         <- Running list of key facts
  user/            <- Per-user knowledge
  sessions/        <- Session summaries
"""
)

# Demonstrate learning across sessions
print("=== Session 1: Learning about GST ===")
r1 = knowledge_agent.invoke(
    {"messages": "Tell me about GST and save key facts to your knowledge base"},
    config={"configurable": {"thread_id": "session-1"}}
)
print(r1["messages"][-1].content[:200])

print("\n=== Session 2: Different thread, but knowledge persists ===")
r2 = knowledge_agent.invoke(
    {"messages": "What do you know about GST?"},
    config={"configurable": {"thread_id": "session-999"}}  # Completely different session!
)
print(r2["messages"][-1].content[:300])
# Agent reads from /knowledge/ which persists via StoreBackend
```

---

## 14.4 Production Memory: SQLite Backend

```python
# chapter_14/sqlite_persistence.py
# Production-ready persistence that survives restarts

# pip install langgraph-checkpoint-sqlite

from langgraph.checkpoint.sqlite import SqliteSaver
from deepagents import create_deep_agent

# SqliteSaver saves state to a SQLite database file
# This survives Python restarts, computer reboots, etc.
DB_PATH = "agent_memory.db"  # Adjust path as needed

with SqliteSaver.from_conn_string(DB_PATH) as checkpointer:
    agent = create_deep_agent(
        model="openai:gpt-4o",
        checkpointer=checkpointer,
        system_prompt="You have persistent memory across restarts."
    )
    
    config = {"configurable": {"thread_id": "persistent-user"}}
    
    # This state is saved to agent_memory.db
    result = agent.invoke(
        {"messages": "Remember: My favorite framework is DeepAgents."},
        config=config
    )
    
    print("Saved to SQLite. Restart Python and run again — it will remember!")
    print(result["messages"][-1].content)

# To verify persistence: run this script again.
# Even after Python restart, the agent will remember the previous conversation.
```

---

## Summary

Three memory types:
1. **Conversation history** (in-context): Available by default, thread-scoped
2. **Checkpointed state** (MemorySaver/SqliteSaver): Multi-call sessions
3. **Cross-session store** (InMemoryStore/SqliteStore): Long-term knowledge

For production: Use SQLite or PostgreSQL for both checkpointer and store.

---

[← Chapter 13](../chapter-13/chapter-13.md) | [Chapter 15 →](../chapter-15/chapter-15.md)
