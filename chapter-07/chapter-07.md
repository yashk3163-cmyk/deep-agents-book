# Chapter 7: Planning Middleware — How Agents Think Step-by-Step

> *"At OpenAI, we ran an experiment: two identical agents, same model, same tools. One had planning, one didn't. On 3-step tasks: 78% completion vs 45%. On 10-step tasks: 61% vs 12%. Planning matters enormously."*

---

## What You'll Learn

- How the `TodoListMiddleware` works internally
- Customizing planning behavior for your domain
- When agents should and shouldn't plan
- Building agents that adapt their plan mid-execution
- Combining planning with reflection (a powerful pattern)

---

## 7.1 How Planning Works

When an agent gets a complex task, the planning middleware prompts it to:
1. Write an initial to-do list (`write_todos`)
2. Execute tasks one by one
3. Update the list as items complete
4. Adapt the list if new information changes the approach

This isn't magic — it's prompt engineering baked into middleware:[cite:14]

```python
# What TodoListMiddleware adds to the system prompt (approximately):
"""
Before starting any task with multiple steps, use the write_todos tool
to create a structured plan. Format your todos as a list of clear,
actionable steps. As you complete each step, update the todo list
to mark it complete. If your plan needs to change based on what you
discover, update the list accordingly.
"""

# And it adds the write_todos tool:
def write_todos(todos: list[dict]) -> str:
    """
    Update your task list. Use before starting complex tasks and
    to track progress as you work.
    
    Each todo item: {"task": str, "status": "pending"|"in_progress"|"done"}
    """
    # Stores the todo list in the agent's state
    ...
```

---

## 7.2 Customizing the Planning Middleware

```python
# chapter_07/custom_planning.py

from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware

# Create an agent with customized planning behavior
agent = create_agent(
    model="openai:gpt-4o",
    middleware=[
        TodoListMiddleware(
            system_prompt="""
Planning Rules:
1. Always create a todo list for tasks with 3+ steps
2. For legal research tasks specifically:
   - Step 1 should always be: 'Identify relevant statute and section'
   - Step 2: 'Search for case law'
   - Step 3: 'Analyze and summarize'
3. Mark todos as 'in_progress' when you start them, 'done' when complete
4. If a step reveals the plan was wrong, update the plan immediately
5. Never proceed to step N+1 until step N is marked 'done'
"""
        )
    ],
    system_prompt="You are a legal research assistant."
)

# Test with a multi-step legal research task
result = agent.invoke({
    "messages": """
    Research the following: What is the current position of courts on 
    whether interest paid on loans for purchase of stocks is deductible 
    under Section 57(iii) of the Income Tax Act?
    """
})

print(result["messages"][-1].content)
```

---

## 7.3 The Reflection Pattern

Reflection is one of the most powerful agent patterns from research at Anthropic and OpenAI: after completing a task, the agent reviews its own work and improves it.

```python
# chapter_07/reflection_agent.py

from dotenv import load_dotenv
load_dotenv()

from deepagents import create_deep_agent

# An agent that automatically reflects on and improves its output
reflective_agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-5-20250929",  # Claude is excellent at reflection
    system_prompt="""
Work Process:
1. Plan your work with write_todos
2. Complete the task
3. Save your first draft to /drafts/first_draft.txt
4. ALWAYS do a reflection pass:
   a. Read your first draft
   b. Ask yourself: Is this accurate? Complete? Well-organized?
   c. Identify 3 specific improvements
   d. Write the improved version
5. Only present the improved version as your final answer

This reflection step is MANDATORY for any response over 200 words.
"""
)

result = reflective_agent.invoke({
    "messages": "Explain what ITAT is and how appeals are filed before it."
})

print(result["messages"][-1].content)
```

---

## 7.4 Adaptive Planning: Changing Plans Mid-Task

One of the most powerful features of deep agents is adapting their plan when they discover unexpected information:

```python
# chapter_07/adaptive_planning.py

from dotenv import load_dotenv
load_dotenv()

from langchain.tools import tool
from deepagents import create_deep_agent

@tool
def check_database(query: str) -> str:
    """Check if specific data exists in our database.
    Returns 'FOUND: <data>' or 'NOT_FOUND'"""
    # Simulate a database - sometimes data exists, sometimes it doesn't
    fake_db = {
        "gst_rate_gold": "3% GST on gold jewelry",
        "gst_rate_electronics": "18% GST on electronics",
    }
    key = query.lower().replace(" ", "_")
    if key in fake_db:
        return f"FOUND: {fake_db[key]}"
    return "NOT_FOUND"

@tool
def fetch_from_api(endpoint: str) -> str:
    """Fetch data from our external API when database has no result."""
    return f"API result for {endpoint}: [simulated data from live API]"

agent = create_deep_agent(
    model="openai:gpt-4o",
    tools=[check_database, fetch_from_api],
    system_prompt="""
Always plan before acting. When you check the database:
- If FOUND: use that data and mark the 'fetch data' todo as done
- If NOT_FOUND: UPDATE your todo list to add a step for fetching from API
This adaptive planning ensures you always get the data one way or another.
"""
)

result = agent.invoke({
    "messages": "What is the GST rate on luxury watches?"
})
# Database won't have this. Agent should:
# 1. Plan: check DB -> use result
# 2. Check DB: NOT_FOUND
# 3. Update plan: add 'fetch from API'
# 4. Call fetch_from_api
# 5. Report result
print(result["messages"][-1].content)
```

---

## Summary

- `TodoListMiddleware` adds the `write_todos` tool and planning instructions
- Customize via `system_prompt` parameter on `TodoListMiddleware`
- The reflection pattern: plan → execute → review → improve → present
- Adaptive planning: agents update their to-do list when discoveries change the approach
- Planning improves task completion dramatically on complex, multi-step work

---

[← Chapter 6](../chapter-06/chapter-06.md) | [Chapter 8 →](../chapter-08/chapter-08.md)
