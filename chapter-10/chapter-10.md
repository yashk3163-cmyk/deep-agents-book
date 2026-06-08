# Chapter 10: Project 1 — Build Your Own Claude Code
## *A Fully Autonomous AI Coding Assistant*

> *"Claude Code is the most impressive coding agent I've used. I know because I helped build some of the patterns it uses. And now I'm going to show you how to build a version of it yourself."*

---

## What You'll Build

A coding agent called **DevAgent** that can:
- Read and understand your codebase
- Write new code files from specifications
- Find and fix bugs
- Refactor code
- Run tests
- Explain what code does
- Create a plan before touching any file

This is not a toy — this is a production-capable coding assistant.

**Estimated build time:** 2-3 hours

---

## 10.1 Architecture Overview

```
User Request: "Add input validation to the login function"
        ↓
[DevAgent - Main Agent]
  ├─ write_todos: [1. Read login function, 2. Identify issues, 3. Write fix, 4. Test]
  ├─ read_file: reads current login function
  ├─ web_search: searches for best practices (optional)
  ├─ write_file: writes improved version
  ├─ run_tests: executes test suite  
  └─ report: explains what was changed and why
        ↓
   Fixed, tested code + explanation
```

---

## 10.2 The Complete Code: DevAgent

```python
# chapter_10/dev_agent.py
# Build Your Own Claude Code - A Complete Coding Agent

from dotenv import load_dotenv
load_dotenv()

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional
from langchain.tools import tool
from deepagents import create_deep_agent
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.backends import FilesystemBackend
from langgraph.checkpoint.memory import MemorySaver

# =============================================================================
# CONFIGURATION
# =============================================================================

# The workspace where DevAgent can read/write code
# IMPORTANT: Point this to your actual project directory
WORKSPACE = os.getcwd()  # Current directory by default
                          # Or set to: "/path/to/your/project"

print(f"DevAgent workspace: {WORKSPACE}")

# =============================================================================
# TOOLS - The hands of DevAgent
# =============================================================================

@tool
def read_code_file(relative_path: str) -> str:
    """Read the contents of a code file.
    
    Use this to understand existing code before modifying it.
    ALWAYS read a file before editing it.
    
    Args:
        relative_path: Path relative to project root (e.g., 'src/main.py', 'tests/test_auth.py')
    
    Returns:
        File contents with line numbers, or error message if file not found
    """
    full_path = Path(WORKSPACE) / relative_path
    
    if not full_path.exists():
        return f"File not found: {relative_path}\nProject structure:\n{get_project_structure.invoke({'max_depth': 2})}"
    
    if not full_path.is_file():
        return f"{relative_path} is a directory, not a file. Use list_files to explore it."
    
    try:
        content = full_path.read_text(encoding='utf-8')
        # Add line numbers for better reference
        lines = content.split('\n')
        numbered = [f"{i+1:4d} | {line}" for i, line in enumerate(lines)]
        return f"File: {relative_path} ({len(lines)} lines)\n\n" + "\n".join(numbered)
    except UnicodeDecodeError:
        return f"Binary file: {relative_path} - cannot read as text"


@tool
def write_code_file(relative_path: str, content: str) -> str:
    """Write or create a code file.
    
    Use this to create new files or completely rewrite existing ones.
    For small edits to existing files, prefer edit_code_file instead.
    
    Args:
        relative_path: Path relative to project root (e.g., 'src/new_module.py')
        content: The complete file content to write
    
    Returns:
        Confirmation message with file path and size
    """
    full_path = Path(WORKSPACE) / relative_path
    
    # Create parent directories if they don't exist
    full_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Back up existing file before overwriting
    if full_path.exists():
        backup_path = full_path.with_suffix(full_path.suffix + '.bak')
        backup_path.write_text(full_path.read_text())
        backup_note = f" (backed up to {backup_path.name})"
    else:
        backup_note = " (new file)"
    
    full_path.write_text(content, encoding='utf-8')
    
    lines = content.count('\n') + 1
    return f"Successfully wrote {relative_path} ({lines} lines){backup_note}"


@tool
def edit_code_file(relative_path: str, old_code: str, new_code: str) -> str:
    """Edit an existing code file by replacing a specific section.
    
    Use this for targeted edits - replacing a function, fixing a bug, etc.
    Prefer this over write_code_file for small changes.
    
    Args:
        relative_path: Path to the file to edit
        old_code: The exact code string to find and replace. Must be exact match.
        new_code: The replacement code
    
    Returns:
        Confirmation or error message
    """
    full_path = Path(WORKSPACE) / relative_path
    
    if not full_path.exists():
        return f"File not found: {relative_path}"
    
    content = full_path.read_text(encoding='utf-8')
    
    if old_code not in content:
        # Help debug the mismatch
        return (f"Could not find the specified code in {relative_path}.\n"
                f"First 200 chars of old_code: {old_code[:200]}\n"
                f"Tip: Use read_code_file to see the exact content, then retry.")
    
    # Make the backup
    backup = full_path.with_suffix(full_path.suffix + '.bak')
    backup.write_text(content)
    
    # Perform the replacement
    new_content = content.replace(old_code, new_code, 1)  # Replace first occurrence only
    full_path.write_text(new_content, encoding='utf-8')
    
    return f"Successfully edited {relative_path} (backup saved as {backup.name})"


@tool
def list_files(relative_path: str = ".", pattern: str = "*") -> str:
    """List files in a directory.
    
    Args:
        relative_path: Directory to list (default: project root)
        pattern: File pattern to filter (e.g., '*.py', '*.txt', '*')
    
    Returns:
        List of files matching the pattern
    """
    full_path = Path(WORKSPACE) / relative_path
    
    if not full_path.exists():
        return f"Directory not found: {relative_path}"
    
    files = sorted(full_path.glob(pattern))
    
    result = [f"Files in {relative_path} matching '{pattern}':"]
    for f in files[:50]:  # Limit to 50 files
        rel = f.relative_to(full_path)
        size = f.stat().st_size if f.is_file() else 0
        type_marker = "/" if f.is_dir() else ""
        result.append(f"  {rel}{type_marker}" + (f" ({size} bytes)" if size else ""))
    
    if len(files) > 50:
        result.append(f"  ... and {len(files) - 50} more")
    
    return "\n".join(result)


@tool
def get_project_structure(max_depth: int = 3) -> str:
    """Get the overall structure of the project as a tree.
    
    Use this at the start of a coding session to understand the project layout.
    
    Args:
        max_depth: How many directory levels to show (default: 3)
    
    Returns:
        Tree structure of the project
    """
    def build_tree(path: Path, depth: int, prefix: str = "") -> list[str]:
        if depth <= 0:
            return [prefix + "...(deeper)"]
        
        items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name))
        lines = []
        
        # Filter out common ignore patterns
        ignore = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', 
                  '.env', '*.pyc', '.DS_Store', 'dist', 'build', '*.bak'}
        items = [i for i in items if i.name not in ignore 
                 and not i.name.endswith('.pyc')]
        
        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            connector = "└── " if is_last else "├── "
            extension = "    " if is_last else "│   "
            
            if item.is_dir():
                lines.append(prefix + connector + item.name + "/")
                lines.extend(build_tree(item, depth - 1, prefix + extension))
            else:
                size = item.stat().st_size
                lines.append(prefix + connector + item.name + f" ({size}b)")
        
        return lines
    
    root = Path(WORKSPACE)
    tree_lines = [f"Project: {root.name}/"]
    tree_lines.extend(build_tree(root, max_depth))
    return "\n".join(tree_lines)


@tool
def run_python_file(relative_path: str, args: str = "") -> str:
    """Run a Python file and return its output.
    
    Use this to test code after writing or modifying it.
    
    Args:
        relative_path: Path to the Python file to run
        args: Command line arguments to pass (space-separated string)
    
    Returns:
        stdout output, or stderr if the file fails
    """
    full_path = Path(WORKSPACE) / relative_path
    
    if not full_path.exists():
        return f"File not found: {relative_path}"
    
    cmd = [sys.executable, str(full_path)] + (args.split() if args else [])
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,  # 30 second timeout
            cwd=WORKSPACE
        )
        
        output = []
        if result.stdout:
            output.append(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            output.append(f"STDERR:\n{result.stderr}")
        if result.returncode != 0:
            output.append(f"Exit code: {result.returncode} (error)")
        else:
            output.append(f"Exit code: 0 (success)")
        
        return "\n".join(output) if output else "(No output)"
    
    except subprocess.TimeoutExpired:
        return f"Timeout: {relative_path} took more than 30 seconds"
    except Exception as e:
        return f"Error running {relative_path}: {e}"


@tool
def run_tests(test_path: str = "tests/") -> str:
    """Run pytest tests and return results.
    
    Use this to verify that changes don't break existing tests.
    
    Args:
        test_path: Path to test file or directory (default: 'tests/')
    
    Returns:
        pytest output with pass/fail counts
    """
    cmd = [sys.executable, "-m", "pytest", test_path, "-v", "--tb=short"]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=WORKSPACE
        )
        output = result.stdout + result.stderr
        return output[-3000:] if len(output) > 3000 else output  # Last 3000 chars
    except subprocess.TimeoutExpired:
        return "Tests timed out after 120 seconds"
    except FileNotFoundError:
        return "pytest not found. Install with: pip install pytest"


@tool
def search_code(query: str, file_pattern: str = "*.py") -> str:
    """Search for a string/pattern across all code files.
    
    Use this to find where something is defined or used across the project.
    
    Args:
        query: The text to search for
        file_pattern: File type to search in (default: '*.py')
    
    Returns:
        List of file:line matches
    """
    root = Path(WORKSPACE)
    matches = []
    
    for filepath in root.rglob(file_pattern):
        # Skip hidden dirs and common non-code dirs
        if any(part.startswith('.') or part in ('venv', '__pycache__', 'node_modules') 
               for part in filepath.parts):
            continue
        
        try:
            content = filepath.read_text(encoding='utf-8', errors='ignore')
            for i, line in enumerate(content.split('\n'), 1):
                if query.lower() in line.lower():
                    rel_path = filepath.relative_to(root)
                    matches.append(f"{rel_path}:{i}: {line.strip()}")
        except Exception:
            pass
    
    if not matches:
        return f"No matches found for '{query}' in {file_pattern} files"
    
    result = [f"Found {len(matches)} match(es) for '{query}':"]
    result.extend(matches[:30])  # Limit to 30 matches
    if len(matches) > 30:
        result.append(f"... and {len(matches) - 30} more matches")
    
    return "\n".join(result)

# =============================================================================
# CREATE THE DEVAGENT
# =============================================================================

checkpointer = MemorySaver()  # Remember conversation history

dev_agent = create_deep_agent(
    model="openai:gpt-4o",  # GPT-4o is excellent for coding tasks
    # For local/private: model="ollama:qwen2.5:14b"
    
    tools=[
        get_project_structure,  # Understand the codebase
        read_code_file,         # Read files
        write_code_file,        # Create/overwrite files
        edit_code_file,         # Targeted edits
        list_files,             # Directory listing
        run_python_file,        # Execute code
        run_tests,              # Run test suite
        search_code,            # Find things in code
    ],
    
    checkpointer=checkpointer,  # Remember across invocations
    
    system_prompt=f"""
You are DevAgent - an expert AI coding assistant. You operate within this project:
{WORKSPACE}

Your Workflow (ALWAYS follow this order):
1. EXPLORE: Use get_project_structure to understand the codebase
2. PLAN: Write a todo list with write_todos before making any changes
3. READ: Always read files with read_code_file before modifying them
4. IMPLEMENT: Write or edit code using the appropriate tool
5. TEST: Run tests or the code to verify your changes work
6. REPORT: Explain what you did and why, in plain English

Code Quality Standards:
- Write clean, documented Python code
- Add docstrings to all functions
- Follow PEP 8 style guidelines
- Include type hints where helpful
- Never remove existing functionality unless explicitly asked
- Create backups are automatic - don't worry about breaking things

File Rules:
- ALWAYS read a file before editing it
- Use edit_code_file for small changes, write_code_file for new files or rewrites
- Never write to files outside {WORKSPACE}

When you encounter errors:
- Read the full error message carefully
- Check the file you just edited
- Search for similar patterns in the codebase
- Explain what went wrong and how you fixed it
"""
)

# =============================================================================
# INTERACTIVE CLI
# =============================================================================

def run_dev_agent():
    """Run DevAgent as an interactive CLI."""
    print("🤖 DevAgent - Your AI Coding Assistant")
    print(f"   Workspace: {WORKSPACE}")
    print("   Commands: 'quit' to exit, 'new' to start fresh session")
    print("-" * 60)
    
    session_id = "devagent-main"
    config = {"configurable": {"thread_id": session_id}}
    
    while True:
        try:
            user_input = input("\n💬 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            
            if user_input.lower() == 'new':
                import uuid
                session_id = f"devagent-{uuid.uuid4().hex[:8]}"
                config = {"configurable": {"thread_id": session_id}}
                print(f"New session started: {session_id}")
                continue
            
            print("\n🤖 DevAgent: Thinking...")
            
            # Stream the response
            final_message = ""
            for chunk in dev_agent.stream(
                {"messages": user_input},
                config=config,
                stream_mode="values"
            ):
                # The last message in the stream is the current state
                messages = chunk.get("messages", [])
                if messages:
                    last = messages[-1]
                    if hasattr(last, 'content') and last.content and last.type == 'ai':
                        final_message = last.content
            
            print(f"\n🤖 DevAgent:\n{final_message}")
        
        except KeyboardInterrupt:
            print("\nInterrupted. Type 'quit' to exit.")
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    run_dev_agent()
```

---

## 10.3 Example Interactions

Here's what a session with DevAgent looks like:

```
💬 You: Show me the structure of this project

🤖 DevAgent: 
Project: my_project/
├── src/
│   ├── auth.py (2,341b)
│   ├── database.py (1,823b)
│   └── utils.py (945b)
├── tests/
│   └── test_auth.py (1,230b)
├── main.py (456b)
└── requirements.txt (123b)

---
💬 You: The login function in src/auth.py has no input validation. Fix it.

🤖 DevAgent:
[Creates todo list: 1. Read auth.py, 2. Identify login function, 3. Add validation, 4. Test]
[Reads src/auth.py]
[Edits the login function to add validation]
[Runs tests]

I've updated the login function in src/auth.py. Here's what I added:
1. Email format validation using regex
2. Password length check (min 8 chars)
3. Input sanitization (strip whitespace)
4. Specific error messages for each validation failure

All 4 existing tests still pass. The backup is at src/auth.py.bak.
```

---

## 10.4 Running DevAgent

```bash
# Make sure you're in your virtual environment
python chapter_10/dev_agent.py

# Or run with a specific project directory:
WORKSPACE=/path/to/your/project python chapter_10/dev_agent.py
```

---

## 10.5 Upgrading to Ollama (Free/Private Version)

```python
# Just change the model line:
dev_agent = create_deep_agent(
    model="ollama:qwen2.5:14b",  # Free, local, private
    tools=[...],
    ...  # Everything else stays the same!
)
```

---

## Summary

You've built a fully functional AI coding assistant with:
- File reading, writing, and editing
- Project structure exploration
- Code search across the project
- Test running
- Planning before acting
- Memory across sessions
- Streaming output

This is architecturally very similar to how Claude Code works — the same four pillars (planning, filesystem, tools, system prompt) in action.

---

[← Chapter 9](../chapter-09/chapter-09.md) | [Chapter 11 →](../chapter-11/chapter-11.md)
