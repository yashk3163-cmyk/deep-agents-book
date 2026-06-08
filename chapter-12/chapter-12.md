# Chapter 12: Project 3 — Build a Deep Research Agent
## *An Agent That Researches Any Topic Like a PhD Student*

> *"Deep Research by OpenAI and Gemini Deep Research are the most impressive AI products of 2024. They work because they loop: search, read, decide if enough, search more. This chapter teaches you to build exactly that."*

---

## What You'll Build

**ResearchAgent** — an autonomous research agent that:
- Takes any research question
- Autonomously decides how many searches to do
- Reads and synthesizes multiple sources
- Produces structured, cited research reports
- Saves reports to the filesystem
- Works with Ollama (free) or cloud models

---

## 12.1 The Research Loop Architecture

```
User Question: "What are the tax implications of crypto in India?"
        ↓
[ResearchAgent]
  ├─ write_todos: [search, synthesize, write report, review]
  ├─ web_search: "crypto tax India 2025"
  ├─ write_file: /research/search_1.txt (raw results)
  ├─ web_search: "virtual digital assets VDA tax India"
  ├─ write_file: /research/search_2.txt
  ├─ web_search: "section 115BBH crypto tax rate"
  ├─ write_file: /research/search_3.txt
  ├─ read_file: /research/search_1.txt  (synthesize all findings)
  ├─ write_file: /report/draft.md (write the report)
  ├─ read_file: /report/draft.md (review)
  └─ edit_file: /report/draft.md (improve)
        ↓
Structured research report with citations
```

---

## 12.2 The Complete Research Agent

```python
# chapter_12/research_agent.py

from dotenv import load_dotenv
load_dotenv()

import os
from datetime import datetime
from typing import Literal
from langchain.tools import tool
from deepagents import create_deep_agent
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.backends import FilesystemBackend  # Save to real disk
from langgraph.checkpoint.memory import MemorySaver

# =============================================================================
# RESEARCH OUTPUT DIRECTORY
# =============================================================================

OUTPUT_DIR = os.path.expanduser("~/research_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Research reports will be saved to: {OUTPUT_DIR}")

# =============================================================================
# RESEARCH TOOLS
# =============================================================================

@tool
def web_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False
) -> str:
    """Search the internet for information.
    
    Use multiple targeted queries to research a topic thoroughly.
    For legal/tax topics, use specific section numbers and terms.
    
    Args:
        query: Specific, targeted search query
        max_results: Number of results (1-10, default 5)
        topic: 'general' for web, 'news' for recent news, 'finance' for financial
        include_raw_content: True to get full page content (slower but more detail)
    
    Returns:
        Formatted search results with titles, URLs, and content snippets
    """
    from tavily import TavilyClient
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    
    results = client.search(
        query,
        max_results=max_results,
        topic=topic,
        include_raw_content=include_raw_content
    )
    
    output = [f"Search: '{query}' | {len(results['results'])} results\n"]
    
    for i, r in enumerate(results["results"], 1):
        output.append(f"[{i}] {r['title']}")
        output.append(f"    URL: {r['url']}")
        output.append(f"    Content: {r['content'][:400]}")
        if include_raw_content and r.get('raw_content'):
            output.append(f"    Full content: {r['raw_content'][:1000]}")
        output.append("")
    
    return "\n".join(output)


@tool
def get_webpage_content(url: str) -> str:
    """Fetch the full content of a specific webpage.
    
    Use this when a search result looks highly relevant and you need
    the complete article or document content.
    
    Args:
        url: The full URL of the webpage to fetch
    
    Returns:
        Full text content of the webpage, or error message
    """
    try:
        import urllib.request
        import html.parser
        
        # Simple HTML stripper
        class HTMLStripper(html.parser.HTMLParser):
            def __init__(self):
                super().__init__()
                self.text = []
            def handle_data(self, data):
                self.text.append(data)
            def get_text(self):
                return ' '.join(self.text)
        
        headers = {'User-Agent': 'Mozilla/5.0 (Research Agent)'}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            html_content = response.read().decode('utf-8', errors='ignore')
        
        stripper = HTMLStripper()
        stripper.feed(html_content)
        text = stripper.get_text()
        
        # Clean up whitespace
        import re
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text[:5000] + ("..." if len(text) > 5000 else "")
    
    except Exception as e:
        return f"Could not fetch {url}: {e}"


@tool
def get_research_date() -> str:
    """Get current date for research report headers."""
    return datetime.now().strftime("%B %d, %Y")


# =============================================================================
# CREATE THE RESEARCH AGENT
# =============================================================================

checkpointer = MemorySaver()

research_agent = create_deep_agent(
    model="openai:gpt-4o",  # Excellent for research synthesis
    # Local alternative: "ollama:qwen2.5:14b"
    
    tools=[
        web_search,
        get_webpage_content,
        get_research_date,
    ],
    
    checkpointer=checkpointer,
    
    middleware=[
        FilesystemMiddleware(
            # Save to REAL disk so reports persist!
            backend=lambda rt: FilesystemBackend(root_dir=OUTPUT_DIR)
        )
    ],
    
    system_prompt="""
You are a meticulous research agent. Your job is to conduct thorough research
and produce high-quality, well-organized research reports.

## RESEARCH PROCESS (Always follow this)

1. PLAN: Use write_todos to create a research plan:
   - List 4-6 specific search queries to cover the topic
   - Plan for synthesis and report writing
   - Plan for review

2. RESEARCH (Execute the plan):
   - Run each planned search query
   - After each search, write key findings to /findings/<query_number>.txt
   - If a result looks very relevant, use get_webpage_content for full content
   - Aim for at least 3-4 searches before moving to synthesis

3. SYNTHESIZE:
   - Read all your findings files
   - Identify key themes, data points, and contradictions
   - Write a synthesis to /synthesis.txt

4. WRITE REPORT:
   - Write your report to /report/report.md
   - Use proper Markdown formatting
   - Report structure:
     # Research Report: [Topic]
     **Date:** [get_research_date()]
     **Searches conducted:** [N]
     
     ## Executive Summary
     [3-4 sentence overview]
     
     ## Key Findings
     [Organized findings with citations as footnotes]
     
     ## Analysis
     [Your synthesis and insights]
     
     ## Sources
     [List of all URLs cited]

5. REVIEW:
   - Read the report back
   - Check: Is it complete? Accurate? Well-organized?
   - Edit any gaps

## RESEARCH STANDARDS
- Never state facts without basis from your searches
- Cite sources: use [Source: URL] inline
- For legal/tax topics: always note the specific statute/section
- Acknowledge uncertainty: use 'appears to be', 'according to [source]'
- Target length: 500-1500 words depending on topic complexity
"""
)

# =============================================================================
# RESEARCH INTERFACE
# =============================================================================

def research(question: str, session_id: str = None) -> str:
    """Conduct research on a question and return the report."""
    
    if session_id is None:
        import uuid
        session_id = f"research-{uuid.uuid4().hex[:8]}"
    
    config = {"configurable": {"thread_id": session_id}}
    
    print(f"\n🔍 Researching: {question}")
    print(f"   Session: {session_id}")
    print(f"   Reports saved to: {OUTPUT_DIR}")
    print("-" * 60)
    
    # Stream progress to show what the agent is doing
    step_count = 0
    for step in research_agent.stream(
        {"messages": question},
        config=config,
        stream_mode="updates"
    ):
        step_count += 1
        if "agent" in step:
            for msg in step["agent"].get("messages", []):
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        args_preview = str(tc['args'])[:80]
                        print(f"  🔧 [{step_count}] {tc['name']}({args_preview}...)")
    
    # Get final result
    result = research_agent.invoke(
        {"messages": "Please provide your final research report now."},
        config=config
    )
    
    report = result["messages"][-1].content
    
    # Save to a dated file
    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    report_file = os.path.join(OUTPUT_DIR, f"report_{date_str}.md")
    with open(report_file, 'w') as f:
        f.write(f"# Research Report\n")
        f.write(f"**Question:** {question}\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(report)
    
    print(f"\n✅ Report saved: {report_file}")
    return report


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Command line: python research_agent.py "Your research question"
        question = " ".join(sys.argv[1:])
    else:
        print("🔬 Deep Research Agent")
        question = input("What would you like to research? ")
    
    report = research(question)
    print("\n" + "=" * 60)
    print("RESEARCH REPORT:")
    print("=" * 60)
    print(report)
```

---

## 12.3 Running the Research Agent

```bash
# Interactive mode
python chapter_12/research_agent.py

# Direct question mode
python chapter_12/research_agent.py "What are the ITAT rulings on Section 14A disallowance in 2024?"

# The report is saved to ~/research_output/report_YYYYMMDD_HHMM.md
```

---

## 12.4 Upgrading to a Fully Local Research Agent

```python
# For completely private, offline research:
research_agent = create_deep_agent(
    model="ollama:qwen2.5:14b",  # Local model
    # Note: web_search still needs internet, but the LLM is local
    # For fully offline: replace web_search with a local document reader
    ...
)
```

---

## Summary

You've built a Deep Research Agent that:
- Plans research with multiple targeted searches
- Saves findings to files to manage context
- Synthesizes multiple sources into a coherent report
- Reviews and improves its own work
- Saves reports to your filesystem

This is architecturally identical to how OpenAI's Deep Research and Gemini's Deep Research work — the "research loop" of search, read, assess, search more.

---

[← Chapter 11](../chapter-11/chapter-11.md) | [Chapter 13 →](../chapter-13/chapter-13.md)
