# Chapter 11: Project 2 — Build Your Own Personal AI Assistant
## *An Agent That Knows You, Remembers You, and Grows With You*

> *"The difference between a useful AI assistant and a great one is memory. An assistant that remembers you don't like bullet points, that you're a tax lawyer in Indore, and that you prefer formal English — that's valuable."*

---

## What You'll Build

**ARIA** (Autonomous Remembering Intelligent Assistant) — a personal AI assistant that:
- Remembers facts about you persistently across sessions
- Learns your preferences from interactions
- Manages your to-do lists and notes
- Answers questions using web search when needed
- Sends email on your behalf (optional)
- Speaks in a style tailored to YOU
- Works with both cloud and local models

---

## 11.1 The Architecture

```
[User] ↔ [ARIA Agent]
              ├─ /memories/profile.txt     (who you are)
              ├─ /memories/preferences.txt  (how you like things)
              ├─ /memories/notes/<topic>.txt (your saved notes)
              ├─ /tasks/todo.txt             (your task list)
              └─ Tools: web_search, email, calendar
```

---

## 11.2 The Complete Code

```python
# chapter_11/aria.py
# ARIA - Your Personal AI Assistant

from dotenv import load_dotenv
load_dotenv()

import os
from datetime import datetime
from langchain.tools import tool
from deepagents import create_deep_agent
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver

# =============================================================================
# PERSISTENT STORAGE SETUP
# =============================================================================
# In production, replace InMemoryStore with:
# from langgraph.store.sqlite import SqliteStore
# store = SqliteStore("aria_memories.db")

store = InMemoryStore()      # Replace with SqliteStore for true persistence!
checkpointer = MemorySaver() # Replace with SqliteSaver for true persistence!

# =============================================================================
# ARIA'S TOOLS
# =============================================================================

@tool
def web_search(query: str, max_results: int = 3) -> str:
    """Search the internet for current information.
    
    Use when the user asks about:
    - Current events or news
    - Prices, rates, statistics
    - Facts you're not certain about
    - Anything that may have changed recently
    """
    from tavily import TavilyClient
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    results = client.search(query, max_results=max_results)
    
    output = []
    for r in results["results"]:
        output.append(f"**{r['title']}**")
        output.append(f"Source: {r['url']}")
        output.append(r['content'][:300])
        output.append("")
    return "\n".join(output)


@tool
def get_datetime() -> str:
    """Get the current date and time. Use when date/time awareness is needed."""
    return datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")


@tool
def add_reminder(task: str, due_date: str = "") -> str:
    """Add a task or reminder.
    
    Args:
        task: Description of the task
        due_date: When it's due (e.g., 'tomorrow', '2025-12-31', 'end of week')
    
    Returns:
        Confirmation that the reminder was saved
    """
    # In production, integrate with Google Calendar, Notion, etc.
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    reminder_line = f"- [{timestamp}] {task}"
    if due_date:
        reminder_line += f" (Due: {due_date})"
    reminder_line += "\n"
    
    # Save to a local file for now
    reminders_file = os.path.expanduser("~/aria_reminders.txt")
    with open(reminders_file, 'a') as f:
        f.write(reminder_line)
    
    return f"Reminder added: {task}" + (f" (Due: {due_date})" if due_date else "")


@tool
def get_reminders() -> str:
    """Get all current reminders and tasks."""
    reminders_file = os.path.expanduser("~/aria_reminders.txt")
    if not os.path.exists(reminders_file):
        return "No reminders found."
    with open(reminders_file, 'r') as f:
        content = f.read()
    return content or "No reminders found."


@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression.
    
    Examples:
    - '15000 * 0.18' for GST calculation
    - '(85000 * 1.18)' for amount with GST
    - '3500000 / 12' for monthly breakdown
    """
    try:
        result = eval(expression, 
                     {"__builtins__": {}}, 
                     {"abs": abs, "round": round, "min": min, "max": max})
        # Format the result nicely for Indian number system
        if isinstance(result, (int, float)):
            return f"{expression} = {result:,.2f}"
        return f"{expression} = {result}"
    except Exception as e:
        return f"Could not calculate '{expression}': {e}"


# =============================================================================
# CREATE ARIA
# =============================================================================

def make_backend(runtime):
    return CompositeBackend(
        default=StateBackend(runtime),
        routes={
            "/memories/": StoreBackend(runtime),  # Persistent across sessions
        }
    )

aria = create_deep_agent(
    model="openai:gpt-4o",  # or "ollama:qwen2.5:14b" for local
    
    tools=[
        web_search,
        get_datetime,
        add_reminder,
        get_reminders,
        calculate,
    ],
    
    store=store,
    checkpointer=checkpointer,
    
    middleware=[
        FilesystemMiddleware(backend=make_backend)
    ],
    
    system_prompt="""
You are ARIA - a personal AI assistant with persistent memory.

## MEMORY PROTOCOL (Follow this EVERY session)

At the START of EVERY conversation:
1. Call ls("/memories/") to check what memories exist
2. If /memories/profile.txt exists: read_file("/memories/profile.txt")
3. If /memories/preferences.txt exists: read_file("/memories/preferences.txt")
4. Use this context to personalize ALL responses

DURING the conversation:
- When the user shares personal info (name, job, location, family, etc.):
  Update /memories/profile.txt immediately
- When you notice preferences (they correct your style, ask for brevity, etc.):
  Update /memories/preferences.txt
- When asked to save notes on a topic:
  Write to /memories/notes/<topic_name>.txt

## MEMORY FILE FORMATS

/memories/profile.txt:
  Name: <full name>
  Profession: <job title and field>
  Location: <city, state, country>
  Organization: <company/firm name>
  Expertise: <areas of expertise>
  Personal: <family, interests, etc.>
  Last updated: <date>

/memories/preferences.txt:
  Communication style: <formal/informal/technical>
  Response length: <brief/detailed>
  Language: <English/Hindi/etc.>
  Format: <bullet points/prose/numbered>
  Domain focus: <primary domain>
  Dislikes: <things to avoid>
  Last updated: <date>

## BEHAVIOR

- Match your communication style to the user's preferences
- Use their name when you know it
- Reference past context: "As you mentioned last time..."
- For calculations, always use the calculate tool
- For current information, always use web_search
- Be genuinely helpful, not just responsive
"""
)

# =============================================================================
# INTERACTIVE INTERFACE
# =============================================================================

def run_aria():
    """Run ARIA as an interactive personal assistant."""
    print("🤖 ARIA - Personal AI Assistant")
    print("   Type 'quit' to exit, 'new session' for a fresh start")
    print("-" * 50)
    
    user_id = "default_user"
    config = {"configurable": {"thread_id": f"aria-{user_id}-main"}}
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            if user_input.lower() == 'new session':
                import uuid
                config = {"configurable": {"thread_id": f"aria-{uuid.uuid4().hex[:8]}"}}
                print("New session started (memories still preserved)")
                continue
            
            result = aria.invoke(
                {"messages": user_input},
                config=config
            )
            
            response = result["messages"][-1].content
            print(f"\nARIA: {response}")
        
        except KeyboardInterrupt:
            print("\nInterrupted.")
            break

if __name__ == "__main__":
    run_aria()
```

---

## 11.3 Example Session

```
You: My name is Yash. I'm a tax advocate in Indore. I specialize in ITAT and CIT(A) appeals.

ARIA: Thank you, Yash! I've saved your profile. As an ITAT and CIT(A) appeals 
specialist in Indore, I'll tailor my responses accordingly. How can I help you today?

You: What's the current penalty rate for late filing under Section 234F?

ARIA: [Searches web]
The penalty under Section 234F for late filing of Income Tax Returns (ITR) is:
- Rs. 5,000 if return is filed after due date but before December 31 of the AY
- Rs. 10,000 if filed after December 31 of the AY
- Reduced to Rs. 1,000 if total income is below Rs. 5 lakhs

---
[Next session - ARIA remembers everything]

You: What do you know about me?

ARIA: Good to see you again, Yash! Here's what I know:
- You're a tax advocate based in Indore, Madhya Pradesh
- You specialize in ITAT (Income Tax Appellate Tribunal) and CIT(A) appeals
- You practice Income Tax law in India
```

---

## 11.4 Making ARIA Truly Persistent

For real persistence across computer restarts, replace the in-memory store:

```python
# Production persistence with SQLite
# pip install langgraph-checkpoint-sqlite
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore  # If available

# These survive computer restarts!
checkpointer = SqliteSaver.from_conn_string("aria.db")
store = SqliteStore("aria_store.db")  # Check LangGraph docs for latest API
```

---

## Summary

You've built ARIA, a personal AI assistant with:
- Persistent memory via the filesystem + StoreBackend
- Personalization based on your profile and preferences
- Web search for current information
- Reminder management
- Financial calculations
- Multi-session memory

---

[← Chapter 10](../chapter-10/chapter-10.md) | [Chapter 12 →](../chapter-12/chapter-12.md)
