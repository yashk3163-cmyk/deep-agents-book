# Chapter 1: Setting Up Your AI Development Environment

> *"A craftsman is only as good as their tools. Before we build agents, we set up the workshop."*

---

## What You'll Learn in This Chapter

- Install Python, pip, and a virtual environment
- Install the DeepAgents library and all dependencies
- Set up API keys for OpenAI and/or Anthropic
- Install Ollama for free, local LLM access
- Run your very first test to confirm everything works

**Estimated reading + setup time:** 30-45 minutes

---

## 1.1 Why a Virtual Environment? (Don't Skip This)

Before we install anything, let's talk about **virtual environments**. If you've done Python before, you may have just run `pip install something` globally. That works, but it creates problems when:

- Project A needs `langchain==0.3.0` but Project B needs `langchain==1.0.0`
- You update one package and break another project
- Your colleague can't reproduce your setup

A virtual environment is like a **separate Python bubble** for each project. Everything installed inside it stays inside it.

> 📝 **SIDE NOTE: What is pip?**
>
> `pip` is Python's package manager. When you run `pip install deepagents`, pip downloads the `deepagents` package from the Python Package Index (PyPI — a giant free repository of Python packages) and installs it on your computer. Think of it like an app store for Python code.

### Creating Your Virtual Environment

**On macOS/Linux:**
```bash
# Navigate to where you want your project
mkdir deep-agents-projects
cd deep-agents-projects

# Create a virtual environment called 'venv'
python3 -m venv venv

# Activate it (you'll see (venv) appear in your terminal prompt)
source venv/bin/activate
```

**On Windows:**
```bash
mkdir deep-agents-projects
cd deep-agents-projects
python -m venv venv
venv\Scripts\activate
```

You should now see `(venv)` at the start of your terminal prompt. This means you're inside the virtual environment. Any `pip install` you run now only affects this project.

> 📝 **SIDE NOTE: What is `python3 -m venv`?**
>
> The `-m` flag tells Python to run a **module** as a script. `venv` is a built-in Python module that creates virtual environments. The `venv` at the end is the folder name you're creating. You can name it anything, but `venv` is the convention.

---

## 1.2 Installing Core Libraries

With your virtual environment active, let's install everything we need:

```bash
# The star of the show
pip install deepagents

# DeepAgents installs langchain and langgraph automatically,
# but let's also install a few useful extras
pip install langchain-openai      # For GPT-4o access
pip install langchain-anthropic   # For Claude access  
pip install langchain-ollama      # For local Ollama models
pip install tavily-python         # Web search tool
pip install python-dotenv         # For managing API keys safely
```

Let's understand what each of these does:

| Package | What It Does |
|---------|-------------|
| `deepagents` | The main framework (pulls in langchain + langgraph) |
| `langchain-openai` | Connector to OpenAI's API (GPT-4o, etc.) |
| `langchain-anthropic` | Connector to Anthropic's API (Claude) |
| `langchain-ollama` | Connector to Ollama (local free models) |
| `tavily-python` | Web search API (agents can search the internet) |
| `python-dotenv` | Loads API keys from a `.env` file securely |

> 📝 **SIDE NOTE: What is LangGraph?**
>
> DeepAgents is built **on top of** LangGraph, which is built **on top of** LangChain. Think of it like layers:
> - **LangChain**: Core building blocks (models, prompts, tools)
> - **LangGraph**: Agent runtime with state, persistence, loops
> - **DeepAgents**: Ready-made "batteries included" agent on top of LangGraph
>
> You don't need to understand LangGraph deeply to use DeepAgents, but knowing it's there helps when you need to debug or customize deeply.

---

## 1.3 Setting Up API Keys

### The `.env` File Pattern (The Right Way)

NEVER hardcode API keys in your Python files like this:

```python
# ❌ NEVER DO THIS
client = openai.OpenAI(api_key="sk-abc123realkey")
```

If you commit that to GitHub, your key is public and will be stolen within minutes by automated bots (this is a real and common problem).

Instead, use a `.env` file:

```bash
# Create a .env file in your project folder
touch .env
```

Open `.env` in any text editor and add your keys:

```bash
# .env file - DO NOT commit this to Git!
OPENAI_API_KEY=sk-your-real-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-real-anthropic-key
TAVILY_API_KEY=tvly-your-tavily-key
```

Now create a `.gitignore` file so Git never uploads your secrets:

```bash
# .gitignore
.env
venv/
__pycache__/
*.pyc
```

### Getting Your API Keys

**OpenAI API Key:**
1. Go to https://platform.openai.com/
2. Create an account
3. Go to "API Keys" in the dashboard
4. Click "Create new secret key"
5. Copy it immediately (shown only once!)

**Anthropic API Key:**
1. Go to https://console.anthropic.com/
2. Create an account
3. Go to "API Keys"
4. Generate a new key

**Tavily API Key (for web search):**
1. Go to https://tavily.com/
2. Sign up for free
3. Your API key is on the dashboard
4. Free tier gives 1,000 searches/month

> 📝 **SIDE NOTE: Do I need to pay?**
>
> **OpenAI**: Requires a paid account, but the cost is tiny. GPT-4o costs about $5 per 1 million input tokens. A typical agent run uses 10,000-50,000 tokens — so about $0.05-$0.25 per run. You can also set spending limits.
>
> **Anthropic**: Same structure. Claude Sonnet is excellent and affordable.
>
> **Free Option (Ollama)**: Chapter 9 covers running everything locally for free. If you don't want to spend any money, skip to Chapter 9 first and come back.

---

## 1.4 Loading API Keys in Python

Create a file called `config.py` in your project:

```python
# config.py
# This file loads our API keys from the .env file

from dotenv import load_dotenv  # Import the dotenv library
import os                        # Import Python's built-in os module

# load_dotenv() reads the .env file and puts all the
# KEY=VALUE pairs into Python's environment variables
load_dotenv()

# os.getenv() retrieves an environment variable by name
# The second argument is a default value if the key isn't found
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# Let's verify the keys loaded (without printing the actual keys!)
if OPENAI_API_KEY:
    print("✅ OpenAI API key loaded")
else:
    print("⚠️  OpenAI API key NOT found - check your .env file")

if ANTHROPIC_API_KEY:
    print("✅ Anthropic API key loaded")
else:
    print("⚠️  Anthropic API key NOT found")

if TAVILY_API_KEY:
    print("✅ Tavily API key loaded")
else:
    print("⚠️  Tavily API key NOT found")
```

Run it:
```bash
python config.py
# Expected output:
# ✅ OpenAI API key loaded
# ✅ Anthropic API key loaded  
# ✅ Tavily API key loaded
```

> 📝 **SIDE NOTE: What are Environment Variables?**
>
> Environment variables are key-value pairs that live in your operating system's memory during a session. They're separate from your Python code. The `os` module lets Python read them. `dotenv` is a convenience library that reads from a `.env` file and sets those environment variables automatically when you call `load_dotenv()`. LangChain and DeepAgents are designed to automatically look for `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` environment variables, so once they're set, you often don't even need to pass them explicitly.

---

## 1.5 Installing Ollama (Free Local LLMs)

Ollama lets you run large language models entirely on your own computer, for free, with no API keys needed.

**Install Ollama:**

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows: Download the installer from https://ollama.ai/
```

**Start Ollama:**
```bash
ollama serve
# Ollama is now running as a local server on http://localhost:11434
```

**Pull your first model (in a new terminal):**
```bash
# Qwen2.5 14B - Excellent for agents, free, runs on 16GB RAM
ollama pull qwen2.5:14b

# Smaller option if you have limited RAM (8GB)
ollama pull qwen2.5:7b

# Llama 3.1 8B - Also great
ollama pull llama3.1:8b
```

Test that Ollama works:
```bash
ollama run qwen2.5:7b "Say hello in exactly 5 words"
# Output: Hello there, how are you?
```

> 📝 **SIDE NOTE: What does "pulling" a model mean?**
>
> LLMs are stored as large files (several gigabytes). "Pulling" a model means downloading that file from Ollama's servers to your computer. Once downloaded, it runs entirely locally — no internet needed. `qwen2.5:14b` means Qwen 2.5 model with 14 billion parameters. More parameters = smarter but slower and heavier. For most agent tasks, 7B or 14B models work well. We'll explore which models work best with DeepAgents in Chapter 9.

---

## 1.6 Verify Everything Works: The Hello World Test

Create `test_setup.py`:

```python
# test_setup.py
# Run this to confirm your entire setup works

print("Testing DeepAgents setup...")
print("=" * 40)

# Test 1: Can we import deepagents?
try:
    from deepagents import create_deep_agent
    print("✅ deepagents imported successfully")
except ImportError as e:
    print(f"❌ deepagents import failed: {e}")
    print("   Fix: pip install deepagents")

# Test 2: Can we import langchain?
try:
    from langchain.agents import create_agent
    print("✅ langchain imported successfully")
except ImportError as e:
    print(f"❌ langchain import failed: {e}")

# Test 3: Can we import dotenv?
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ python-dotenv working")
except ImportError as e:
    print(f"❌ dotenv import failed: {e}")
    print("   Fix: pip install python-dotenv")

# Test 4: Do we have at least one API key or Ollama?
import os
openai_key = os.getenv("OPENAI_API_KEY", "")
anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")

if openai_key or anthropic_key:
    print("✅ At least one cloud API key found")
else:
    print("⚠️  No cloud API keys — that's OK if you're using Ollama")

# Test 5: Is Ollama running?
import urllib.request
try:
    with urllib.request.urlopen("http://localhost:11434") as response:
        print("✅ Ollama is running on localhost:11434")
except:
    print("⚠️  Ollama not running — run 'ollama serve' first if using local models")

print("=" * 40)
print("Setup test complete!")
print("If you see all ✅ marks, you're ready to build agents.")
```

Run it:
```bash
python test_setup.py
```

---

## 1.7 Project Folder Structure

Here's the folder structure we'll use throughout the book:

```
deep-agents-projects/
├── .env                    ← API keys (NEVER commit)
├── .gitignore              ← Keeps .env out of Git
├── venv/                   ← Virtual environment
├── config.py               ← Shared config loader
├── chapter_03/
│   └── first_agent.py
├── chapter_10/
│   └── claude_code_clone.py
├── chapter_11/
│   └── personal_assistant.py
├── chapter_12/
│   └── research_agent.py
└── chapter_13/
    └── research_team.py
```

---

## Summary

You've set up:
- ✅ A Python virtual environment
- ✅ DeepAgents and all required libraries
- ✅ API key management with `.env`
- ✅ Ollama for local/free models
- ✅ A working test to verify everything

**In the next chapter**, we'll understand what LLMs actually are — how they work, what tokens are, and why this matters when building agents.

---

## Practice Questions

1. What does `source venv/bin/activate` do, and why do we do it before installing packages?
2. Why should you never put your API key directly in your Python code?
3. What is Ollama, and how does it differ from using the OpenAI API?
4. In `config.py`, what does `os.getenv("OPENAI_API_KEY", "")` do if the key is not set?

**Answers in Appendix B.**

---

[← Introduction](../intro/introduction.md) | [Chapter 2 →](../chapter-02/chapter-02.md)
