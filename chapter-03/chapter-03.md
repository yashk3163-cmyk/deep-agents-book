# Chapter 3: Your First Agent — Hello, DeepAgents!

> *"The best way to understand a framework is to build something with it immediately. We'll explore why each line works the way it does after we see it working."*

---

## What You'll Learn

- Create your first deep agent with `create_deep_agent`
- Understand every parameter of `create_deep_agent`
- Run your agent and read its output
- Understand the agent loop: how the agent "thinks"
- See the four built-in tools every deep agent has automatically

---

## 3.1 The Simplest Deep Agent

Here it is. The simplest working deep agent:

```python
# chapter_03/first_agent.py

# Step 1: Load our API keys from .env
from dotenv import load_dotenv
load_dotenv()  # This reads .env and sets environment variables

# Step 2: Import the one function we need from deepagents
from deepagents import create_deep_agent
# create_deep_agent is the main factory function in DeepAgents
# A "factory function" is a function that creates and returns an object

# Step 3: Create the agent
agent = create_deep_agent(
    model="openai:gpt-4o"  # Which LLM to use as the agent's brain
    # That's the minimum! Everything else has defaults.
)

# Step 4: Give the agent a task
result = agent.invoke({
    "messages": "What is 5 + 3? Show your work step by step."
})

# Step 5: Print the final response
# The agent's last message is always the final answer
print(result["messages"][-1].content)
```

Run it:
```bash
python first_agent.py
```

You'll see something like:
```
Let me work through this step by step:

1. I'll write my approach to /plan.txt first...
[writes a todo list]
2. Start: 5
3. Add 3 to 5
4. Result: 8

The answer is 8.
```

Notice something: the agent **automatically wrote a to-do list** before answering a simple math question. That's the planning middleware at work — it's on by default.

---

## 3.2 Understanding Every Parameter

Now let's use ALL the parameters of `create_deep_agent`:

```python
# chapter_03/full_agent.py

from dotenv import load_dotenv
load_dotenv()

from deepagents import create_deep_agent

agent = create_deep_agent(
    # ------- REQUIRED -------
    
    model="openai:gpt-4o",
    # Which LLM powers the agent. Format: "provider:model_name"
    # Options: "openai:gpt-4o", "anthropic:claude-sonnet-4-5-20250929",
    #          "ollama:qwen2.5:14b", "openai:gpt-4o-mini"
    
    # ------- OPTIONAL (but important) -------
    
    tools=[],  
    # A list of Python functions that the agent can call.
    # We'll add real tools in Chapter 5. Empty list means
    # the agent only uses its built-in tools (filesystem, planning, subagents).
    
    system_prompt="You are a helpful assistant. Be concise.",
    # Additional instructions appended to the built-in deep agent prompt.
    # Note: DeepAgents has its own detailed system prompt already.
    # Your system_prompt is ADDED to it, not replacing it.
    # This is where you define the agent's personality and domain.
    
    # ------- ADVANCED (skip for now) -------
    
    # subagents=[],          # Specialist agents to delegate to (Chapter 8)
    # middleware=[],         # Custom middleware (Chapter 7)
    # store=None,            # LangGraph store for cross-session memory (Chapter 14)
    # checkpointer=None,     # For saving/resuming agent state (Chapter 14)
)

# The invoke method starts the agent running
result = agent.invoke({
    # "messages" is the required key
    # It can be a string (convenience) or a list of message dicts
    "messages": "Write a haiku about Python programming."
})

# result is a dict. The most important key is "messages"
# result["messages"] is a list of all messages in the conversation
# result["messages"][-1] is the last message (the final answer)
final_message = result["messages"][-1]
print("Final answer:", final_message.content)

# You can also see the full conversation history
print("\n--- Full conversation ---")
for i, msg in enumerate(result["messages"]):
    # msg.type is 'human', 'ai', 'tool', or 'system'
    print(f"[{i}] {msg.type.upper()}: {str(msg.content)[:100]}...")
```

---

## 3.3 The Agent Loop: How DeepAgents Thinks

This is the most important concept in the book. Understanding the agent loop is understanding everything.

Here's what happens when you call `agent.invoke()`:

```
START
  │
  ▼
[1] LLM receives: system_prompt + user message + conversation history
  │
  ▼
[2] LLM decides: Do I answer directly, or do I call a tool?
  │
  ├── If ANSWER: Return text → DONE
  │
  └── If TOOL CALL: →
              │
              ▼
            [3] Execute the tool (write_file, internet_search, etc.)
              │
              ▼
            [4] Tool result added to conversation history
              │
              ▼
            [5] Go back to step [1] with updated history
              │
              ▼
            [2] LLM decides again...
              (Loop continues until LLM answers directly)
```

Let's trace this with actual code:

```python
# chapter_03/trace_agent_loop.py
# This version lets us see EVERY step the agent takes

from dotenv import load_dotenv
load_dotenv()

from deepagents import create_deep_agent
from langchain_core.messages import AIMessage, ToolMessage

agent = create_deep_agent(
    model="openai:gpt-4o",
    system_prompt="You are a helpful assistant."
)

# Use stream() instead of invoke() to see steps as they happen
for step in agent.stream(
    {"messages": "Plan and write 3 facts about the moon."},
    stream_mode="updates"  # Stream intermediate updates
):
    # Each 'step' is a dict with information about what just happened
    
    if "agent" in step:
        # The LLM just produced a response
        messages = step["agent"]["messages"]
        for msg in messages:
            if isinstance(msg, AIMessage):
                if msg.tool_calls:
                    # The AI decided to call a tool
                    for tc in msg.tool_calls:
                        print(f"\n🔧 TOOL CALL: {tc['name']}")
                        print(f"   Args: {tc['args']}")
                else:
                    # The AI produced a final answer
                    print(f"\n🤖 FINAL ANSWER:")
                    print(msg.content)
    
    elif "tools" in step:
        # A tool just returned a result
        messages = step["tools"]["messages"]
        for msg in messages:
            if isinstance(msg, ToolMessage):
                print(f"
✅ TOOL RESULT ({msg.name}):")
                print(f"   {str(msg.content)[:200]}")
```

When you run this, you'll see the agent:
1. Call `write_todos` to plan its work
2. Then produce the actual content
3. Possibly call `write_file` to save intermediate work
4. Return the final answer

> 📝 **SIDE NOTE: What is `stream()` vs `invoke()`?**
>
> `invoke()` runs the entire agent and returns when it's done. You wait for the full result. `stream()` is a generator that yields updates as they happen. For long-running agents, streaming lets you see progress in real-time and build responsive UIs. Both produce the same final result.

---

## 3.4 The Four Built-in Tools Every Deep Agent Has

When you create a deep agent with `create_deep_agent()`, it automatically gets four built-in tools. These are always available, even if you pass `tools=[]`:[cite:14]

### Tool 1: `write_todos`
Purpose: The agent writes and updates its to-do list
```
Agent: "I need to research, write, and review. Let me track this."
  → Calls write_todos({"todos": ["Research topic", "Write draft", "Review"]})
  → Updates internal todo list
  → Now the agent tracks its progress
```

### Tools 2-5: Filesystem Tools (`ls`, `read_file`, `write_file`, `edit_file`)
Purpose: The agent can manage files in a virtual filesystem
```
Agent: "I'll save my intermediate research to a file so I don't lose it."
  → Calls write_file({"path": "/research/notes.txt", "content": "..."})
  → Saves to virtual filesystem (lives in agent's state by default)
  → Later, calls read_file({"path": "/research/notes.txt"}) to retrieve it
```

### Tool 6: `task` (Subagent tool)
Purpose: Delegate complex subtasks to a subagent
```
Agent: "This research task is complex. I'll delegate to a subagent."
  → Calls task({"agent": "general-purpose", "task": "Research X in detail"})
  → A new agent instance runs with the subtask
  → Returns the result without polluting main agent's context
```

Let's see these tools being used:

```python
# chapter_03/observe_builtin_tools.py

from dotenv import load_dotenv
load_dotenv()

from deepagents import create_deep_agent

agent = create_deep_agent(
    model="openai:gpt-4o",
    system_prompt="""You are a note-taking assistant.
When given a topic, always:
1. Write a todo list first
2. Save your notes to a file
3. Read the file back to verify it was saved
4. Then give your final answer"""
)

result = agent.invoke({
    "messages": "Write 3 key facts about Python's GIL (Global Interpreter Lock)"
})

# Now let's look at ALL messages to see what tools were called
for msg in result["messages"]:
    msg_type = type(msg).__name__
    if msg_type == "AIMessage":
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"🔧 Agent called: {tc['name']}({list(tc['args'].keys())})")
        else:
            print(f"🤖 Agent final answer: {msg.content[:100]}...")
    elif msg_type == "ToolMessage":
        print(f"✅ Tool '{msg.name}' returned: {str(msg.content)[:80]}...")
```

---

## 3.5 Configuring Threads (Session Management)

Deep agents can maintain state across multiple invocations using **thread IDs**:

```python
# chapter_03/thread_management.py

from dotenv import load_dotenv
load_dotenv()

from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver

# MemorySaver keeps state in-memory between calls
# In production, you'd use a database (Chapter 14)
checkpointer = MemorySaver()

agent = create_deep_agent(
    model="openai:gpt-4o",
    checkpointer=checkpointer,  # Enable state persistence
    system_prompt="You are a helpful assistant. Remember what the user tells you."
)

# Thread ID groups related conversations together
# Same thread_id = same "session" / same memory
config = {"configurable": {"thread_id": "user-session-001"}}

# First message in this session
result1 = agent.invoke(
    {"messages": "My name is Yash and I'm a lawyer in Indore, India."},
    config=config  # Pass the thread config
)
print("Response 1:", result1["messages"][-1].content)

# Second message - same thread, so agent "remembers" the first message
result2 = agent.invoke(
    {"messages": "What is my name and what do I do?"},
    config=config  # Same thread_id = agent sees previous messages
)
print("Response 2:", result2["messages"][-1].content)
# Should say: "Your name is Yash, you're a lawyer in Indore, India."

# Different thread - no memory of previous conversation
config_new = {"configurable": {"thread_id": "different-user-999"}}
result3 = agent.invoke(
    {"messages": "What is my name?"},
    config=config_new
)
print("Response 3 (new thread):", result3["messages"][-1].content)
# Will say: "I don't know your name, you haven't told me."
```

> 📝 **SIDE NOTE: What is a Thread?**
>
> In LangGraph (which DeepAgents runs on), a "thread" is a named sequence of runs. Thread IDs are just strings you choose. All runs with the same thread ID share the same checkpointed state. This lets you build applications where the same user can continue conversations across multiple sessions. Without a checkpointer, each `invoke()` call starts fresh — no memory of previous calls.

---

## 3.6 Your First Agent: Complete Working Example

```python
# chapter_03/first_complete_agent.py
# A complete, well-commented first agent

from dotenv import load_dotenv
load_dotenv()

from deepagents import create_deep_agent

# Create a general-purpose assistant agent
assistant = create_deep_agent(
    model="openai:gpt-4o",
    
    system_prompt="""You are a knowledgeable assistant with expertise in technology.
    
When answering questions:
- Always plan your response before writing it (use write_todos)
- If the answer is long, save sections to files as you write them
- Be thorough but concise
- Format your final answer in Markdown"""
)

print("Agent created! Let's ask it something complex.")
print("-" * 50)

# Ask a multi-part question that will exercise the planning and filesystem tools
result = assistant.invoke({
    "messages": """
    Please explain three things:
    1. What is LangChain?
    2. What is LangGraph?
    3. What is DeepAgents?
    And explain how they relate to each other.
    """
})

# Print only the final answer
print(result["messages"][-1].content)

print("-" * 50)
print(f"Total messages in conversation: {len(result['messages'])}")
print("(That number includes all tool calls and responses in the agent loop)")
```

---

## Summary

- `create_deep_agent()` creates a fully-equipped autonomous agent
- The agent loop: LLM calls tools, gets results, loops until it answers
- Every deep agent has planning, filesystem, and subagent tools built-in
- `stream()` lets you observe the agent thinking step-by-step
- Thread IDs enable memory across multiple invocations (with a checkpointer)

---

## Practice Questions

1. What are the 6 built-in tools every deep agent gets automatically?
2. What is the difference between `invoke()` and `stream()`?
3. If you run `agent.invoke()` twice without a checkpointer, does the second call remember the first? Why?
4. In `result["messages"][-1]`, what does `-1` as a list index do in Python?

**Answers in Appendix B.**

---

[← Chapter 2](../chapter-02/chapter-02.md) | [Chapter 4 →](../chapter-04/chapter-04.md)
