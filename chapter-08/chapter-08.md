# Chapter 8: Subagents — Building Teams of Agents

> *"The biggest mistake I see builders make is trying to cram everything into one agent. Nature solved complex tasks with specialization. Your agent architecture should too."*

---

## What You'll Learn

- The subagent architecture and when to use it
- Creating specialized subagents with `SubAgentMiddleware`
- Wrapping LangGraph graphs as subagents
- The `general-purpose` built-in subagent
- Designing supervisor-worker architectures
- Context isolation: why subagents keep the main agent clean

---

## 8.1 The Subagent Mental Model

Think of a consulting firm:
- **Managing Partner (Main Agent)**: Understands the client, plans the project, coordinates
- **Research Analyst (Subagent)**: Deep-dives into data, returns summaries
- **Technical Writer (Subagent)**: Writes polished documents, returns finished sections
- **Financial Analyst (Subagent)**: Handles numbers and calculations

Each specialist works in their own "room" (isolated context). The managing partner doesn't get distracted by all the raw research data — they get clean summaries from specialists.

This is exactly how DeepAgents' subagent system works.[cite:35]

---

## 8.2 Creating Your First Subagent

```python
# chapter_08/basic_subagents.py

from dotenv import load_dotenv
load_dotenv()

from langchain.tools import tool
from langchain.agents import create_agent
from deepagents.middleware.subagents import SubAgentMiddleware

# Tools that ONLY the research subagent will have
@tool
def web_search(query: str) -> str:
    """Search the internet for information."""
    from tavily import TavilyClient
    import os
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    results = client.search(query, max_results=3)
    return "\n".join([f"{r['title']}: {r['content'][:200]}" 
                      for r in results["results"]])

# Tools that ONLY the calculator subagent will have
@tool
def precise_calculate(expression: str) -> str:
    """Perform precise mathematical calculations."""
    import decimal
    decimal.getcontext().prec = 50  # High precision arithmetic
    try:
        result = eval(expression, {"__builtins__": {}, "Decimal": decimal.Decimal})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

# Create the main agent with two specialized subagents
main_agent = create_agent(
    model="openai:gpt-4o",
    middleware=[
        SubAgentMiddleware(
            default_model="openai:gpt-4o",
            default_tools=[],  # Main agent has no direct tools - it only delegates!
            subagents=[
                {
                    "name": "researcher",
                    # This description is what the main LLM reads to decide
                    # when to use this subagent
                    "description": "Expert internet researcher. Use when you need "
                                   "current information, facts, news, or data from the web.",
                    "system_prompt": """You are a thorough internet researcher.
                    Always search at least 2-3 times with different queries
                    to ensure comprehensive coverage. Return a detailed
                    structured summary with key facts and sources.""",
                    "tools": [web_search],
                    "model": "openai:gpt-4o",
                },
                {
                    "name": "calculator",
                    "description": "Precise mathematical calculator. Use for any arithmetic, "
                                   "financial calculations, percentages, or formulas.",
                    "system_prompt": "You are a precise mathematical calculator. "
                                     "Show your work. Always use precise_calculate for computations.",
                    "tools": [precise_calculate],
                    "model": "openai:gpt-4o-mini",  # Cheaper model for simple math
                }
            ]
        )
    ],
    system_prompt="""You are a coordinator. For any research needs, delegate to the 'researcher' subagent.
    For any calculations, delegate to the 'calculator' subagent.
    Synthesize the results into a coherent final answer."""
)

# The main agent will automatically:
# 1. Use task("researcher", ...) to get facts
# 2. Use task("calculator", ...) for math
# 3. Combine results into a clean final answer
result = main_agent.invoke({
    "messages": "What is the current GST rate on restaurant food, "
                "and how much GST would apply to a bill of Rs. 4,750?"
})
print(result["messages"][-1].content)
```

---

## 8.3 The Built-in General-Purpose Subagent

Every deep agent comes with a "general-purpose" subagent available at all times. This subagent has the same model and instructions as the main agent, but runs in an **isolated context**.[cite:14]

```python
# chapter_08/general_purpose_subagent.py

from dotenv import load_dotenv
load_dotenv()

from deepagents import create_deep_agent
from langchain.tools import tool

@tool
def web_search(query: str) -> str:
    """Search the internet."""
    from tavily import TavilyClient
    import os
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    return str(client.search(query, max_results=3)["results"][:2])

agent = create_deep_agent(
    model="openai:gpt-4o",
    tools=[web_search],
    system_prompt="""
For complex research tasks with multiple subtopics:
Use the task('general-purpose', ...) to delegate individual subtopics.
Each subagent call gets its own clean context window.
This keeps your context clean when doing multi-topic research.
"""
)

# This will cause the agent to delegate subtopics to the general-purpose subagent
result = agent.invoke({
    "messages": """
    I need a comprehensive comparison of:
    1. Section 44AD (presumptive taxation for business)
    2. Section 44ADA (presumptive taxation for professionals)
    3. Section 44AE (for transport businesses)
    
    Cover: eligibility, turnover limits, and tax computation method for each.
    """
})
print(result["messages"][-1].content)
```

---

## 8.4 Using a LangGraph Graph as a Subagent

For maximum customization, you can wrap any LangGraph compiled graph as a subagent:[cite:35]

```python
# chapter_08/langgraph_subagent.py

from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from deepagents.middleware.subagents import SubAgentMiddleware, CompiledSubAgent
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from typing import TypedDict, Annotated
import operator

# Define the state for our custom subagent graph
class ResearchState(TypedDict):
    messages: Annotated[list, operator.add]  # Messages accumulate
    search_count: int                          # Track how many searches done
    findings: list[str]                        # Accumulated research findings

# Build a custom LangGraph research workflow
def build_research_graph():
    from langchain_openai import ChatOpenAI
    
    llm = ChatOpenAI(model="gpt-4o-mini")  # Cheaper model for subagent
    
    def research_step(state: ResearchState) -> ResearchState:
        """Perform one research iteration."""
        messages = state["messages"]
        # This is simplified - in practice you'd call actual search tools
        last_message = messages[-1].content if messages else ""
        response = llm.invoke(
            [{"role": "system", "content": "Research this topic thoroughly."},
             {"role": "user", "content": last_message}]
        )
        return {
            "messages": [AIMessage(content=response.content)],
            "search_count": state.get("search_count", 0) + 1,
            "findings": state.get("findings", []) + [response.content[:200]]
        }
    
    def should_continue(state: ResearchState) -> str:
        """Decide if we need more research."""
        if state.get("search_count", 0) >= 2:  # Max 2 research iterations
            return "done"
        return "research_more"
    
    # Build the graph
    workflow = StateGraph(ResearchState)
    workflow.add_node("research", research_step)
    workflow.set_entry_point("research")
    workflow.add_conditional_edges(
        "research",
        should_continue,
        {"research_more": "research", "done": END}
    )
    
    return workflow.compile()

# Wrap it as a DeepAgents subagent
research_graph = build_research_graph()
research_subagent = CompiledSubAgent(
    name="deep_researcher",
    description="A specialized multi-step research agent. Use for complex research "
                "requiring multiple iterations of searching and synthesis.",
    runnable=research_graph  # The compiled LangGraph graph
)

# Create main agent with the custom graph subagent
main_agent = create_agent(
    model="openai:gpt-4o",
    middleware=[
        SubAgentMiddleware(
            default_model="openai:gpt-4o",
            default_tools=[],
            subagents=[research_subagent]  # Our custom graph
        )
    ],
    system_prompt="For research tasks, always use the deep_researcher subagent."
)

result = main_agent.invoke({
    "messages": "Research the key provisions of Section 263 of the Income Tax Act."
})
print(result["messages"][-1].content)
```

---

## 8.5 Supervisor Architecture Pattern

For complex multi-agent systems, use the Supervisor pattern:

```
User Request
    ↓
[SUPERVISOR AGENT]
    ├── [RESEARCHER] — Gathers information
    ├── [ANALYST]   — Processes and analyzes
    ├── [WRITER]    — Drafts the output
    └── [REVIEWER]  — Quality checks the output
    ↓
 Final Output to User
```

We'll build this complete architecture in Chapter 13: The Research Team Project.

---

## Summary

- Subagents provide context isolation: the main agent's context stays clean
- Create subagents with custom tools, models, and system prompts
- The general-purpose subagent is always available for isolated deep-dives
- Custom LangGraph graphs can be wrapped as subagents via `CompiledSubAgent`
- Use the Supervisor pattern for complex multi-step workflows

---

[← Chapter 7](../chapter-07/chapter-07.md) | [Chapter 9 →](../chapter-09/chapter-09.md)
