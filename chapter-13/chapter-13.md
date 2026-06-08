# Chapter 13: Project 4 — Build a Research Team with Multiple Agents
## *A Supervisor-Worker Multi-Agent System Using Ollama + Python + DeepAgents*

> *"The best research I've seen from AI systems comes not from one very smart agent, but from several specialized agents that coordinate. This is true at Anthropic, at Palantir, and it's what we'll build here."*

---

## What You'll Build

A **Research Team** of 4 specialized agents:

| Agent | Role | Tools |
|-------|------|-------|
| **Coordinator** | Supervisor, plans the project, synthesizes final report | No direct tools — delegates to all others |
| **Searcher** | Internet research specialist | web_search, get_webpage |
| **Analyst** | Data analysis and synthesis | calculate, read_files |
| **Writer** | Report writing specialist | write_file, edit_file |

All agents run on **Ollama** (free, local) — no API costs for this project!

---

## 13.1 The Multi-Agent Workflow

```
User: "Research and write a report on GST compliance for small businesses in India"
        ↓
[COORDINATOR AGENT]
  1. Plans the research project
  2. Delegates to Searcher: "Find GST compliance requirements, thresholds, filing deadlines"
        ↓
  [SEARCHER AGENT] - runs in isolation
    - Searches 5+ times with different queries
    - Returns: Detailed research summary
  
  3. Delegates to Analyst: "Analyze these findings, identify key thresholds and dates"
        ↓
  [ANALYST AGENT] - runs in isolation
    - Processes the research
    - Calculates penalties, thresholds
    - Returns: Structured analysis
  
  4. Delegates to Writer: "Write a professional report based on research and analysis"
        ↓
  [WRITER AGENT] - runs in isolation
    - Writes formatted report
    - Saves to /final_report.md
    - Returns: Complete report
  
  5. Coordinator reviews and presents final report
```

---

## 13.2 The Complete Research Team

```python
# chapter_13/research_team.py
# Multi-Agent Research Team using Ollama + DeepAgents

from dotenv import load_dotenv
load_dotenv()

import os
from datetime import datetime
from langchain.tools import tool
from langchain.agents import create_agent
from deepagents import create_deep_agent
from deepagents.middleware.subagents import SubAgentMiddleware
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.backends import FilesystemBackend
from langgraph.checkpoint.memory import MemorySaver

# =============================================================================
# CONFIGURATION
# Use Ollama for free local inference!
# Change to "openai:gpt-4o" or "anthropic:claude-sonnet-4-5-20250929" for cloud
# =============================================================================

DEFAULT_MODEL = "ollama:qwen2.5:14b"   # Free local model
FAST_MODEL = "ollama:qwen2.5:7b"       # Faster for simple tasks

OUTPUT_DIR = os.path.expanduser("~/research_team_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# SHARED TOOLS
# =============================================================================

@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the internet for current information. Use specific, targeted queries."""
    from tavily import TavilyClient
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    results = client.search(query, max_results=max_results)
    output = [f"Results for: {query}\n"]
    for i, r in enumerate(results["results"], 1):
        output.append(f"[{i}] {r['title']} ({r['url']})")
        output.append(f"    {r['content'][:300]}\n")
    return "\n".join(output)


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression for precise calculations."""
    try:
        result = eval(expression, {"__builtins__": {}}, {"abs": abs, "round": round})
        return f"{expression} = {result:,.2f}" if isinstance(result, float) else f"{expression} = {result}"
    except Exception as e:
        return f"Calculation error: {e}"


@tool
def get_date() -> str:
    """Get current date for report headers."""
    return datetime.now().strftime("%B %d, %Y")


@tool
def save_to_file(filename: str, content: str) -> str:
    """Save content to the output directory.
    Args:
        filename: Filename (e.g., 'report.md', 'analysis.txt')
        content: Text content to save
    """
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, 'w') as f:
        f.write(content)
    return f"Saved to {filepath} ({len(content)} chars)"


@tool
def read_saved_file(filename: str) -> str:
    """Read a previously saved file from the output directory."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(filepath):
        return f"File not found: {filename}"
    with open(filepath, 'r') as f:
        return f.read()


# =============================================================================
# BUILD THE RESEARCH TEAM
# =============================================================================

checkpointer = MemorySaver()

# Create the Coordinator (main agent) with specialized subagents
coordinator = create_agent(
    model=DEFAULT_MODEL,
    
    middleware=[
        SubAgentMiddleware(
            default_model=DEFAULT_MODEL,
            default_tools=[],
            subagents=[
                {
                    "name": "searcher",
                    "description": (
                        "Internet research specialist. Use this agent to find current information, "
                        "news, statistics, laws, and facts. Tell it exactly what to research and "
                        "how many searches to do. It will return a comprehensive research summary."
                    ),
                    "system_prompt": """
You are a meticulous internet researcher. When given a research task:
1. Break it into 4-6 specific search queries
2. Run EACH search using web_search
3. Compile all findings into a comprehensive, structured summary
4. Include all relevant facts, numbers, dates, and URLs
5. Organize by theme/category
Be thorough. Cover all aspects of the topic.
""",
                    "tools": [web_search, get_date],
                    "model": DEFAULT_MODEL,
                },
                {
                    "name": "analyst",
                    "description": (
                        "Data analyst and synthesizer. Use when you have research findings "
                        "that need to be processed, calculated, compared, or structured. "
                        "Give it the raw research and ask for specific analysis."
                    ),
                    "system_prompt": """
You are a precise data analyst. When given research findings:
1. Identify key quantitative data (numbers, percentages, thresholds, dates)
2. Perform any necessary calculations using the calculate tool
3. Compare and contrast different pieces of information
4. Identify patterns, trends, and key insights
5. Structure your analysis clearly with headers and bullet points
6. Flag any contradictions or gaps in the research
""",
                    "tools": [calculate, read_saved_file],
                    "model": DEFAULT_MODEL,
                },
                {
                    "name": "writer",
                    "description": (
                        "Professional report writer. Use when you have research and analysis "
                        "ready and need a polished, well-formatted report. "
                        "Give it the research, analysis, and any formatting requirements."
                    ),
                    "system_prompt": """
You are an expert technical and professional writer. When writing a report:
1. Write in clear, professional language
2. Use proper Markdown formatting (headers, bullets, tables)
3. Organize logically: Executive Summary → Findings → Analysis → Recommendations → Sources
4. Include all relevant data points with proper attribution
5. Make it actionable: concrete takeaways and next steps
6. Save the final report using save_to_file
7. Target length: 800-1500 words
""",
                    "tools": [save_to_file, get_date],
                    "model": DEFAULT_MODEL,
                }
            ]
        ),
        FilesystemMiddleware(
            backend=lambda rt: FilesystemBackend(root_dir=OUTPUT_DIR)
        )
    ],
    
    system_prompt=f"""
You are the Coordinator of a research team. You have three specialized agents:
- **searcher**: For internet research and finding information
- **analyst**: For processing, calculating, and synthesizing data
- **writer**: For creating polished, formatted reports

Your Workflow for Any Research Request:
1. Plan with write_todos: [delegate to searcher] → [delegate to analyst] → [delegate to writer] → [review]
2. Delegate research: task('searcher', 'Research [specific aspects] of [topic]')
3. Delegate analysis: task('analyst', 'Analyze these findings: [paste searcher results]')
4. Delegate writing: task('writer', 'Write a professional report on [topic]. Research: [findings]. Analysis: [analysis]')
5. Review the saved report and present key highlights to the user

You do NOT do research yourself. You coordinate specialists.
Reports are saved to: {OUTPUT_DIR}
""",
    checkpointer=checkpointer
)

# =============================================================================
# RUN THE RESEARCH TEAM
# =============================================================================

def run_research_team(question: str) -> str:
    """Run the multi-agent research team on a question."""
    
    session_id = f"team-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    config = {"configurable": {"thread_id": session_id}}
    
    print(f"\n👥 Research Team Starting")
    print(f"   Topic: {question}")
    print(f"   Model: {DEFAULT_MODEL} (Ollama - Free!)")
    print(f"   Output: {OUTPUT_DIR}")
    print("-" * 60)
    
    # Stream to show what's happening
    print("\n[Coordinator]: Planning and delegating...")
    
    result = coordinator.invoke(
        {"messages": question},
        config=config
    )
    
    final_response = result["messages"][-1].content
    
    # Check if a report was saved
    report_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.md')]
    if report_files:
        latest = sorted(report_files)[-1]
        print(f"\n📚 Report saved: {os.path.join(OUTPUT_DIR, latest)}")
    
    return final_response


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        print("👥 Multi-Agent Research Team")
        print(f"   Using: {DEFAULT_MODEL} (Ollama)")
        print("")
        question = input("Research topic: ")
    
    report = run_research_team(question)
    print("\n" + "=" * 60)
    print("COORDINATOR SUMMARY:")
    print("=" * 60)
    print(report)
```

---

## 13.3 Running the Research Team

```bash
# Make sure Ollama is running first!
ollama serve &

# Run the team on a legal research topic
python chapter_13/research_team.py "GST compliance requirements for small businesses under Rs 40 lakh turnover in India"

# Or run interactively
python chapter_13/research_team.py
```

---

## 13.4 Scaling the Team

You can add more specialist agents:

```python
# Add a legal researcher agent
{
    "name": "legal_researcher",
    "description": "Specialized in Indian law. Use for legal sections, case law, and tribunal decisions.",
    "system_prompt": "You specialize in Indian tax law. Always cite section numbers. Research ITAT decisions.",
    "tools": [web_search],
    "model": DEFAULT_MODEL,
},

# Add a fact-checker agent
{
    "name": "fact_checker",
    "description": "Verifies facts by cross-referencing multiple sources.",
    "system_prompt": "For each fact, search at least 2 sources to verify. Flag contradictions.",
    "tools": [web_search],
    "model": FAST_MODEL,  # Cheaper/faster for verification
}
```

---

## Summary

You've built a complete multi-agent research team:
- Coordinator (Supervisor) delegates to specialized agents
- Searcher conducts comprehensive internet research
- Analyst processes and structures the findings
- Writer produces a polished final report
- All running locally on Ollama — free and private

This is the same pattern used in production at major AI companies for complex, multi-step knowledge work.

---

[← Chapter 12](../chapter-12/chapter-12.md) | [Chapter 14 →](../chapter-14/chapter-14.md)
