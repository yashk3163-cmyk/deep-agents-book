# Building Custom AI Agents with DeepAgents
## *A Practical, Code-First Guide to the LangChain DeepAgents Framework*

> **Inspired by "Automate the Boring Stuff with Python" — every concept explained, every line of code justified, and 4 complete real-world projects.**

---

## 📖 About This Book

This is a complete, hands-on guide to building custom AI agents using **DeepAgents** — the autonomous agent framework introduced by LangChain. Written for someone who knows basic Python and has heard of LLMs, but has never built an agent before.

By the end of this book, you will have built:
1. **DevAgent** — Your own Claude Code (AI coding assistant)
2. **ARIA** — A personal AI assistant that remembers you across sessions
3. **ResearchAgent** — A Deep Research agent that writes cited reports
4. **Research Team** — A multi-agent system with Supervisor + Specialists (Ollama, free!)

---

## 📋 Table of Contents

### Part I: Foundations

| Chapter | Title | Key Concepts |
|---------|-------|--------------|
| [Intro](./intro/intro.md) | Why Agents? What is DeepAgents? | Agent vs chatbot, the planning loop |
| [Ch. 1](./chapter-01/chapter-01.md) | Setup and Your First Agent | Python env, API keys, `create_deep_agent()` |
| [Ch. 2](./chapter-02/chapter-02.md) | How LLMs Think | Tokens, temperature, context window |
| [Ch. 3](./chapter-03/chapter-03.md) | Your First Real Agent | `invoke()`, `stream()`, built-in tools |
| [Ch. 4](./chapter-04/chapter-04.md) | Writing Custom Tools | `@tool` decorator, docstrings, type hints |
| [Ch. 5](./chapter-05/chapter-05.md) | The Filesystem Middleware | `ls`, `read_file`, `write_file`, `edit_file` |

### Part II: Advanced Features

| Chapter | Title | Key Concepts |
|---------|-------|--------------|
| [Ch. 6](./chapter-06/chapter-06.md) | The Agent Loop & Planning | `write_todos`, agent loop internals, streaming |
| [Ch. 7](./chapter-07/chapter-07.md) | Memory & Persistence | `MemorySaver`, `SqliteSaver`, checkpoints |
| [Ch. 8](./chapter-08/chapter-08.md) | Subagents — Building Teams | `SubAgentMiddleware`, context isolation |
| [Ch. 9](./chapter-09/chapter-09.md) | Local Agents with Ollama | Free, private, offline agents with Qwen2.5 |

### Part III: Projects

| Chapter | Title | What You Build |
|---------|-------|---------------|
| [Ch. 10](./chapter-10/chapter-10.md) | **Project 1: Build Your Own Claude Code** | DevAgent — Full coding assistant |
| [Ch. 11](./chapter-11/chapter-11.md) | **Project 2: Personal AI Assistant** | ARIA — Remembering personal assistant |
| [Ch. 12](./chapter-12/chapter-12.md) | **Project 3: Deep Research Agent** | ResearchAgent — Cited research reports |
| [Ch. 13](./chapter-13/chapter-13.md) | **Project 4: Multi-Agent Research Team** | 4-agent team on Ollama (free!) |

### Part IV: Production

| Chapter | Title | Key Concepts |
|---------|-------|--------------|
| [Ch. 14](./chapter-14/chapter-14.md) | Memory Architecture Deep Dive | 3 types of memory, SQLite persistence |
| [Ch. 15](./chapter-15/chapter-15.md) | Deploying to Production | FastAPI, Docker, Railway, LangSmith |

### Appendices

| Appendix | Title |
|----------|-------|
| [Appendix A](./appendix-a/appendix-a.md) | Complete DeepAgents API Reference |
| [Appendix B](./appendix-b/appendix-b.md) | Practice Answers & Troubleshooting |

---

## 🚀 Quick Start

```bash
# Clone this repository
git clone https://github.com/yashk3163-cmyk/deep-agents-book
cd deep-agents-book

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate    # Windows

# Install dependencies
pip install deepagents langchain langchain-openai langchain-anthropic
pip install langchain-ollama tavily-python python-dotenv
pip install langgraph fastapi uvicorn

# Set up your API keys
cp .env.example .env
# Edit .env with your keys

# Run your first agent (Chapter 3)
python chapter-03/first_agent.py
```

---

## 🔧 Requirements

- Python 3.11+
- One of: OpenAI API key, Anthropic API key, OR Ollama installed locally
- (Optional) Tavily API key for web search capabilities
- (Optional) LangSmith API key for observability

```
# .env file template
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
TAVILY_API_KEY=tvly-...
LANGCHAIN_API_KEY=lsv2-...  # For LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=deep-agents-book
```

---

## 📦 Key Technologies

| Technology | Role | Cost |
|-----------|------|------|
| **DeepAgents** | Agent framework | Free (open source) |
| **LangChain** | LLM interface layer | Free (open source) |
| **LangGraph** | Agent state machine | Free (open source) |
| **Ollama** | Local LLM runner | Free (local) |
| **Qwen2.5:14b** | Best local agent model | Free |
| **GPT-4o** | Best cloud model | ~$5-15 per book |
| **Tavily** | Web search API | Free tier available |
| **LangSmith** | Agent observability | Free tier available |

---

## 📝 Side Notes Index

Every chapter includes **Side Notes** on underlying concepts:

- 📝 What is a token?
- 📝 What is temperature in an LLM?
- 📝 What is a context window?
- 📝 What is function/tool calling?
- 📝 What is an API?
- 📝 What is JSON?
- 📝 What is a Python decorator (`@tool`)?
- 📝 What is a docstring?
- 📝 What are type hints?
- 📝 What is a REST API?
- 📝 What is Docker?
- 📝 What is a virtual environment?
- 📝 What is an environment variable?

---

## 🇮🇳 India-Specific Examples

Because real learning uses real examples, the book includes India-relevant use cases:
- GST calculation agents
- Income Tax Act section lookups
- ITAT research agents  
- Hindi/English bilingual assistant examples
- Legal document drafting with AI

---

## 📄 License

MIT License — use freely, build commercially, share openly.

---

*Written in the style of "Automate the Boring Stuff with Python" by Al Sweigart — practical, complete, and respectful of the reader's time.*
