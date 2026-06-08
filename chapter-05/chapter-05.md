# Chapter 5: Tools — The Hands of Your Agent

> *"An agent without tools is just a chatbot with planning anxiety. Tools are what let it actually DO things in the world."*

---

## What You'll Learn

- What tools are and how the LLM decides to use them
- How to write your own Python functions as tools
- Using the `@tool` decorator and type hints
- Writing good tool docstrings (the most important part!)
- Built-in tool categories: search, filesystem, code execution
- Connecting to MCP (Model Context Protocol) servers for extra tools

---

## 5.1 What is a Tool?

A tool is a Python function that the agent can call. That's the entire concept.

When the LLM decides it needs to perform an action (search the web, read a file, calculate something), it generates a "tool call" — a structured request to call a specific function with specific arguments.

The DeepAgents framework handles:
1. Showing the LLM what tools are available (via their descriptions)
2. Parsing the LLM's tool call request
3. Calling your Python function with the right arguments
4. Returning the result to the LLM as a ToolMessage

You just write the Python function.

---

## 5.2 Writing Your First Tool

```python
# chapter_05/basic_tools.py

from langchain.tools import tool  # The @tool decorator
from typing import Optional        # For optional parameters

# The simplest possible tool
@tool
def get_current_time() -> str:
    """Get the current date and time. Use this when you need to know what time it is."""
    # The docstring above is what the LLM reads to decide if/when to use this tool.
    # Make it clear and specific!
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# A tool with parameters
@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together and return the result."""
    return a + b

# A tool with optional parameters
@tool
def format_currency(
    amount: float,
    currency: str = "INR",
    decimal_places: int = 2
) -> str:
    """Format a number as currency.
    
    Args:
        amount: The monetary amount to format
        currency: Currency code (default: INR for Indian Rupee)
        decimal_places: Number of decimal places (default: 2)
    
    Returns:
        Formatted currency string like '1,23,456.78 INR'
    """
    # Indian number formatting: lakhs, crores
    if currency == "INR":
        # Indian formatting uses 2-digit grouping after first 3
        formatted = f"{amount:,.{decimal_places}f}"
    else:
        formatted = f"{amount:,.{decimal_places}f}"
    return f"{formatted} {currency}"

# Now use these tools in an agent
from deepagents import create_deep_agent
from dotenv import load_dotenv
load_dotenv()

agent = create_deep_agent(
    model="openai:gpt-4o",
    tools=[get_current_time, add_numbers, format_currency],  # Pass the tools here
    system_prompt="Use the available tools when relevant to answer questions accurately."
)

result = agent.invoke({
    "messages": "What time is it, and what is 1234567.89 formatted as Indian Rupees?"
})
print(result["messages"][-1].content)
```

---

## 5.3 The Secret: Writing Great Docstrings

The docstring is the most important part of a tool. It's what the LLM reads to decide:
1. Does this tool exist for my current need?
2. What parameters do I need to pass?
3. What will I get back?

**Bad docstring (LLM won't know when to use it):**
```python
@tool
def search(q: str) -> dict:
    """Search."""
    ...
```

**Good docstring (LLM knows exactly when and how to use it):**
```python
@tool
def internet_search(
    query: str,
    max_results: int = 5,
    search_type: str = "general"
) -> str:
    """Search the internet for current information.
    
    Use this tool when you need:
    - Current news or recent events (after your training cutoff)
    - Specific facts you're not certain about
    - Recent prices, statistics, or data
    - Information about specific companies, people, or places
    
    Do NOT use for:
    - General knowledge questions you already know
    - Mathematical calculations
    
    Args:
        query: The search query. Be specific and use keywords.
               Example: 'India GDP 2025' not 'economy'
        max_results: Number of results to return (1-10, default 5)
        search_type: 'general' for web, 'news' for recent news only
    
    Returns:
        String with search results including titles, URLs, and snippets
    """
    from tavily import TavilyClient
    import os
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    results = client.search(query, max_results=max_results, topic=search_type)
    
    # Format results as a readable string
    output = []
    for r in results["results"]:
        output.append(f"Title: {r['title']}")
        output.append(f"URL: {r['url']}")
        output.append(f"Content: {r['content'][:300]}")
        output.append("---")
    
    return "\n".join(output)
```

---

## 5.4 Real-World Tool Examples

### Tool: Read a PDF File
```python
@tool
def read_pdf(file_path: str) -> str:
    """Extract text content from a PDF file.
    
    Use this to read PDF documents, legal files, reports, or any PDF.
    The extracted text may lose some formatting but will contain all text content.
    
    Args:
        file_path: Absolute path to the PDF file on the local filesystem
    
    Returns:
        Extracted text from the PDF
    """
    import pdfplumber  # pip install pdfplumber
    
    text_pages = []
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                text_pages.append(f"--- Page {page_num + 1} ---")
                text_pages.append(text)
    
    return "\n".join(text_pages)
```

### Tool: Run Python Code
```python
@tool
def run_python_code(code: str) -> str:
    """Execute a Python code snippet and return the output.
    
    Use this for:
    - Mathematical calculations that need actual computation
    - Data processing and transformation
    - String manipulation
    - Generating structured data
    
    Do NOT use for:
    - Code that accesses the network (use internet_search instead)
    - Code that reads/writes files outside the sandbox
    
    Args:
        code: Valid Python code to execute. Must be complete, runnable code.
              Include print() statements for any values you want to see.
    
    Returns:
        Standard output of the code, or error message if it fails
    """
    import subprocess
    import sys
    
    # Run in a subprocess for isolation
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=10  # 10 second timeout
    )
    
    if result.returncode == 0:
        return result.stdout or "(No output)"
    else:
        return f"Error: {result.stderr}"
```

### Tool: Send an Email
```python
@tool
def send_email(
    to_address: str,
    subject: str,
    body: str
) -> str:
    """Send an email to the specified address.
    
    Use ONLY when the user explicitly requests to send an email.
    Always confirm the recipient and content before sending.
    
    Args:
        to_address: Recipient email address
        subject: Email subject line
        body: Email body text (plain text, no HTML)
    
    Returns:
        Confirmation message with 'sent' or error description
    """
    import smtplib
    from email.mime.text import MIMEText
    import os
    
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    sender = os.getenv("SMTP_EMAIL")
    password = os.getenv("SMTP_PASSWORD")
    
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to_address
    
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(sender, password)
        server.send_message(msg)
    
    return f"Email sent to {to_address} with subject '{subject}'"
```

---

## 5.5 Connecting MCP Servers (Bonus Tools)

MCP (Model Context Protocol) is a standard for connecting external tool servers to AI agents. Many tools (GitHub, Slack, Notion, etc.) provide MCP servers:

```python
# chapter_05/mcp_tools.py
# Using MCP to connect to external tool servers

from langchain_mcp import MCPToolkit
from deepagents import create_deep_agent
from dotenv import load_dotenv
load_dotenv()

# Connect to a GitHub MCP server
# (The MCP server runs as a separate process)
github_toolkit = MCPToolkit.from_command(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-github"],
    env={"GITHUB_TOKEN": "your_github_token"}
)

# Get all tools from the MCP server
github_tools = github_toolkit.get_tools()
print(f"GitHub MCP provides {len(github_tools)} tools:")
for tool in github_tools[:5]:
    print(f"  - {tool.name}: {tool.description[:60]}")

# Create an agent with GitHub tools
agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-5-20250929",
    tools=github_tools,
    system_prompt="You are a GitHub assistant. Help users manage their repositories."
)
```

> 📝 **SIDE NOTE: What is MCP (Model Context Protocol)?**
>
> MCP is an open standard (developed by Anthropic) for connecting AI models to external tools and data sources. Instead of writing custom Python code for every tool (GitHub API, Notion API, Slack API), you connect to a pre-built MCP server that exposes those tools. LangChain and DeepAgents support MCP natively via the LangChain Academy course Deep Research agent.[cite:22] It's like a standardized plugin system for AI agents.

---

## 5.6 Complete Agent with Multiple Tools

```python
# chapter_05/multi_tool_agent.py

from dotenv import load_dotenv
load_dotenv()

import os
from typing import Literal
from langchain.tools import tool
from tavily import TavilyClient
from deepagents import create_deep_agent

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the internet for current information. Use for facts, news, prices."""
    results = tavily.search(query, max_results=max_results)
    output = []
    for r in results["results"]:
        output.append(f"[{r['title']}]({r['url']})")
        output.append(r['content'][:200])
        output.append("")
    return "\n".join(output)

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression. Use for any arithmetic or math.
    Example: '(100 * 1.18) + 500' for a GST calculation."""
    try:
        result = eval(expression, {"__builtins__": {}, "abs": abs, "round": round})
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"

@tool
def get_date() -> str:
    """Get today's date. Use when date-aware responses are needed."""
    from datetime import datetime
    return datetime.now().strftime("%A, %B %d, %Y")

# Create a research + calculation agent
agent = create_deep_agent(
    model="openai:gpt-4o",
    tools=[web_search, calculate, get_date],
    system_prompt="""You are a research and analysis assistant.
Always use web_search for current information.
Use calculate for any math.
Use get_date when time-sensitive."""
)

result = agent.invoke({
    "messages": "What is the current GST rate on gold jewelry in India, "
                "and how much GST would be charged on a purchase of Rs. 2,50,000?"
})
print(result["messages"][-1].content)
```

---

## Summary

- Tools are Python functions decorated with `@tool`
- The docstring is what the LLM reads — make it detailed and specific
- Pass tools to `create_deep_agent()` via the `tools=[]` parameter
- Use MCP servers for pre-built integrations (GitHub, Slack, Notion, etc.)
- Good tool design = clear name + specific docstring + sensible return type

---

[← Chapter 4](../chapter-04/chapter-04.md) | [Chapter 6 →](../chapter-06/chapter-06.md)
