# Chapter 4: The Four Pillars of Deep Agents

> *"Claude Code, Manus, and Deep Research — the most impressive AI agents of 2024-25 — all share exactly four architectural features. Once you understand these four pillars, you understand all of them."*

---

## What You'll Learn

- Why simple LLM loops fail on complex tasks
- The four pillars: Planning, Filesystem, Subagents, System Prompt
- How each pillar solves a specific problem
- How DeepAgents implements each pillar
- How to think about agent architecture

---

## 4.1 The Problem with Shallow Agents

Imagine asking a simple LLM agent: *"Research the top 5 AI companies, compare their products, write a detailed report with citations, save it as a PDF, and email it to me."*

A simple agent (LLM in a loop with tools) will:
- Try to do everything in one massive output
- Forget what it was doing after a few tool calls
- Run out of context window halfway through
- Lose track of what's been done and what hasn't
- Give up or produce incomplete work

This is why they're called "shallow" agents — they can't dive deep on complex, multi-step work.

LangChain's research found that ALL successful deep agents — Claude Code, Manus, Deep Research — solved this with the same four patterns.[cite:19]

---

## 4.2 Pillar 1: Planning (The To-Do List)

### The Problem It Solves
Without planning, an agent jumps straight into action. It starts doing Step 3 before completing Step 1. It forgets what it was supposed to do. It can't check its own progress.

### How It Works
Before tackling a complex task, the agent is prompted (and trained) to call `write_todos` to create an explicit task list:

```python
# What the agent internally does when given a complex task:

# 1. Agent receives task: "Research and write a report on AI trends"

# 2. Agent calls write_todos FIRST:
write_todos({
    "todos": [
        {"id": 1, "task": "Search for top AI trends 2025", "status": "pending"},
        {"id": 2, "task": "Search for AI market size data", "status": "pending"},
        {"id": 3, "task": "Write introduction section", "status": "pending"},
        {"id": 4, "task": "Write main body with data", "status": "pending"},
        {"id": 5, "task": "Write conclusion", "status": "pending"},
        {"id": 6, "task": "Review and format", "status": "pending"}
    ]
})

# 3. Agent completes task 1, updates list:
write_todos({
    "todos": [
        {"id": 1, "task": "Search for top AI trends 2025", "status": "done"},
        {"id": 2, "task": "Search for AI market size data", "status": "in_progress"},
        ...
    ]
})
```

This is **exactly** what Claude Code does. When you ask it to refactor a large codebase, the first thing it shows you is a numbered list of files it plans to touch.[cite:14]

### The `TodoListMiddleware`

```python
# chapter_04/planning_demo.py

from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware

# Create an agent with ONLY the planning middleware
# (no filesystem, no subagents) so we can isolate this pillar
agent = create_agent(
    model="openai:gpt-4o",
    middleware=[
        TodoListMiddleware(
            # Optional: customize the planning instructions
            system_prompt="""Before starting any task with more than 2 steps,
            use write_todos to plan your approach. 
            Update the todo list as you complete each step."""
        )
    ]
)

result = agent.invoke({
    "messages": "Count from 1 to 5, then multiply each number by 3, then sum the results."
})

print(result["messages"][-1].content)
```

You'll see the agent:
1. Call `write_todos` with 3 tasks: count, multiply, sum
2. Execute each step
3. Produce the final answer: 1\*3 + 2\*3 + 3\*3 + 4\*3 + 5\*3 = 45

> 📝 **SIDE NOTE: What is Middleware?**
>
> Middleware is software that sits "in the middle" between two things. In DeepAgents, middleware adds capabilities to an agent without modifying its core logic. It's like adding a plugin. `TodoListMiddleware` adds planning capabilities. `FilesystemMiddleware` adds file I/O. `SubAgentMiddleware` adds delegation. You can add, remove, or customize any middleware independently.

---

## 4.3 Pillar 2: Filesystem (The External Memory)

### The Problem It Solves
LLM context windows are limited. A research task might involve reading 20 web pages totaling 100,000 tokens. You can't keep all of that in the conversation. The filesystem lets the agent write intermediate results to "files" and read them back selectively.

### How It Works

The agent gets four filesystem tools:[cite:14]

| Tool | What it does |
|------|--------------|
| `ls` | List files in the filesystem |
| `read_file` | Read a file (or part of it) |
| `write_file` | Create a new file |
| `edit_file` | Modify an existing file |

```python
# chapter_04/filesystem_demo.py

from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from deepagents.middleware.filesystem import FilesystemMiddleware

# Create an agent with ONLY filesystem middleware
agent = create_agent(
    model="openai:gpt-4o",
    middleware=[FilesystemMiddleware()]
)

# Give the agent a task that requires saving intermediate work
result = agent.invoke({
    "messages": """
    Do the following:
    1. Write the names of 5 programming languages to /languages.txt
    2. Read the file back to confirm it was saved
    3. Edit the file to add 2 more languages
    4. Tell me what's in the file
    """
})

print(result["messages"][-1].content)

# We can also inspect the virtual filesystem from outside the agent
# The filesystem lives in the agent's state
if "filesystem" in result:
    print("\nFiles in agent's virtual filesystem:")
    for path, content in result["filesystem"].items():
        print(f"  {path}: {len(content)} chars")
```

### Short-term vs Long-term Filesystem

By default, the filesystem is **ephemeral** — it lives only for the current thread. This is the `StateBackend`.

For persistent memory across sessions, use `StoreBackend`:[cite:57]

```python
# chapter_04/persistent_filesystem.py

from deepagents import create_deep_agent
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.store.memory import InMemoryStore

# Create a store (in production, use PostgreSQL or Redis instead)
store = InMemoryStore()

# Configure a COMPOSITE backend:
# - Most files: ephemeral (StateBackend) - fast, thread-local
# - /memories/ folder: persistent (StoreBackend) - survives across threads
def make_backend(runtime):
    return CompositeBackend(
        default=StateBackend(runtime),          # Default: ephemeral
        routes={
            "/memories/": StoreBackend(runtime) # Exception: persistent
        }
    )

agent = create_deep_agent(
    model="openai:gpt-4o",
    store=store,
    middleware=[
        FilesystemMiddleware(
            backend=make_backend,  # Use our custom backend
            custom_tool_descriptions={
                "write_file": "Save important information to /memories/ for long-term storage.",
                "read_file": "Read previously saved information from /memories/."
            }
        )
    ],
    system_prompt="When the user tells you something important, save it to /memories/facts.txt"
)

from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()
agent_with_memory = create_deep_agent(
    model="openai:gpt-4o",
    store=store,
    checkpointer=checkpointer,
    system_prompt="""Save important user facts to /memories/user_facts.txt.
    Read from /memories/ when the user asks about themselves."""
)

config = {"configurable": {"thread_id": "session-1"}}
result = agent_with_memory.invoke(
    {"messages": "I'm a tax lawyer in Indore. Save this for later."},
    config=config
)

config2 = {"configurable": {"thread_id": "session-2"}}  # Different session!
result2 = agent_with_memory.invoke(
    {"messages": "What do you know about me?"},
    config=config2
)
print(result2["messages"][-1].content)
# Should recall the user is a tax lawyer in Indore, from session-1
```

> 📝 **SIDE NOTE: What is a Backend in DeepAgents?**
>
> A backend is the storage system that powers the agent's filesystem. DeepAgents ships with four backends:
> - `StateBackend`: Stores in LangGraph state (thread-scoped, RAM)
> - `StoreBackend`: Stores in LangGraph's BaseStore (cross-session, database)
> - `FilesystemBackend`: Stores on your real disk (for local development)
> - `CompositeBackend`: Routes different paths to different backends
>
> In production at companies like Anthropic, agents use database-backed stores so that agent "memories" persist across restarts and scale across multiple servers.[cite:57]

---

## 4.4 Pillar 3: Subagents (The Delegation System)

### The Problem It Solves
Complex tasks require deep focus. But deep focus fills the context window. If the main agent tries to do everything itself, its context gets cluttered with thousands of tokens of intermediate results, making it less effective at the final task.

Subagents solve this: the main agent delegates subtasks to subagents, gets back a clean summary, and keeps its own context clean.

### The Subagent Architecture

```
Main Agent (Supervisor)
  │
  ├── calls task("research", "Research quantum computing")
  │         │
  │    Research Subagent runs in isolated context
  │    (does 20 searches, reads 10 pages, summarizes)
  │    Returns: "Here's a 500-word summary..."
  │
  ├── calls task("writing", "Write intro based on summary")
  │         │
  │    Writing Subagent runs in isolated context
  │    (writes draft, revises, formats)
  │    Returns: "Here's the introduction..."
  │
  └── Main agent assembles final output
      (Context only contains summaries, not all the work)
```

```python
# chapter_04/subagents_demo.py

from dotenv import load_dotenv
load_dotenv()

from langchain.tools import tool
from langchain.agents import create_agent
from deepagents.middleware.subagents import SubAgentMiddleware

# Define a tool for one of the subagents
@tool  # This decorator turns a Python function into a LangChain tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression safely."""
    # In real code, use a safe math evaluator, not eval()!
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

# Create the main agent with a specialized subagent
agent = create_agent(
    model="openai:gpt-4o",
    middleware=[
        SubAgentMiddleware(
            default_model="openai:gpt-4o",
            default_tools=[],
            subagents=[
                {
                    "name": "calculator",
                    "description": "A subagent that can perform mathematical calculations.",
                    "system_prompt": "Use the calculate tool to compute the answer.",
                    "tools": [calculate],  # Only this subagent has the math tool
                    "model": "openai:gpt-4o-mini",  # Cheaper model for simple math
                }
            ]
        )
    ],
    system_prompt="For any math, delegate to the calculator subagent."
)

result = agent.invoke({
    "messages": "What is (144 * 3.14159) + (2 ** 10) - 99?"
})

print(result["messages"][-1].content)
```

> 📝 **SIDE NOTE: What is `@tool` decorator?**
>
> The `@tool` decorator (from `langchain.tools`) transforms a regular Python function into a LangChain "Tool" object. When you decorate a function with `@tool`, LangChain reads the function's name, docstring, and type hints to automatically generate a description that the LLM uses to decide when to call the function. The docstring is extremely important — it's what the LLM reads to understand what the tool does.

---

## 4.5 Pillar 4: The System Prompt (The Agent's Character)

### The Problem It Solves
A generic LLM doesn't know when to plan, when to write files, when to delegate, or how to structure its work. The system prompt is the "operating manual" that tells the agent exactly how to behave.

### DeepAgents' Built-in Prompt

DeepAgents ships with a battle-tested system prompt inspired by Claude Code's prompt. You don't see it (it's internal), but it includes instructions like:
- "Before starting a complex task, use write_todos to plan"
- "Write intermediate results to files to manage context"
- "Use subagents for tasks that need deep focus"
- "When in doubt, break work into smaller pieces"

Your `system_prompt` parameter is APPENDED to this built-in prompt, adding your domain-specific instructions:[cite:19]

```python
# chapter_04/system_prompt_demo.py

from dotenv import load_dotenv
load_dotenv()

from deepagents import create_deep_agent

# Example: A tax law research agent (relevant to our reader!)
tax_agent = create_deep_agent(
    model="openai:gpt-4o",
    system_prompt="""You are an expert in Indian tax law, specializing in:
- Income Tax Act 1961 and 2025
- GST and IGST regulations
- ITAT (Income Tax Appellate Tribunal) procedures
- Tax assessment orders and appeals

When researching tax topics:
1. Always cite the specific section numbers (e.g., "Section 80C", "Rule 8D")
2. Note whether the law is under ITA 1961 or the new ITA 2025
3. Save research notes to /tax_research/current_topic.txt
4. Flag any recent amendments or circulars you know about
5. If asked about case law, structure as: Citation | Tribunal | Key Holding

You are cautious: always note that this is research assistance,
not formal legal advice."""
)

result = tax_agent.invoke({
    "messages": "What are the conditions for disallowance under Section 14A of the Income Tax Act?"
})

print(result["messages"][-1].content)
```

---

## 4.6 How the Four Pillars Work Together: A Visual

```
User: "Research AI trends and write a 5-page report"
          │
          ▼
    ┌────────────────────────────────────────────┐
    │           DEEP AGENT (Main)              │
    │                                          │
    │  ⭐ PILLAR 4: System Prompt              │
    │  "I'm a research expert. Plan first."   │
    │                                          │
    │  ⭐ PILLAR 1: Planning                   │
    │  write_todos([search, research, write])  │
    │                                          │
    │  ⭐ PILLAR 3: Subagents                  │
    │  task("searcher", "Find AI trends")      │
    │    └── Search Subagent runs...           │
    │    └── Returns: 500-word summary         │
    │                                          │
    │  ⭐ PILLAR 2: Filesystem                 │
    │  write_file("/report/section1.txt", ...) │
    │  write_file("/report/section2.txt", ...) │
    │  read_file("/report/section1.txt")       │
    │  └── Assemble final report              │
    └────────────────────────────────────────────┘
          │
          ▼
    User gets: Complete 5-page report
```

---

## Summary

The four pillars of deep agents:
1. **Planning (TodoList)**: Write a to-do list before acting; track progress
2. **Filesystem**: Write intermediate results to files; read them back selectively
3. **Subagents**: Delegate complex subtasks; keep main context clean
4. **System Prompt**: Detailed instructions for when and how to use pillars 1-3

`create_deep_agent()` gives you all four by default. The next chapters dive deep into each one.

---

## Practice Questions

1. Why does the filesystem middleware help with the context window problem?
2. What is the main benefit of using subagents instead of having one agent do everything?
3. In the `SubAgentMiddleware` example, why was `gpt-4o-mini` used for the calculator subagent?
4. Describe in your own words what the `@tool` decorator does.

**Answers in Appendix B.**

---

[← Chapter 3](../chapter-03/chapter-03.md) | [Chapter 5 →](../chapter-05/chapter-05.md)
