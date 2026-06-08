# Chapter 2: LLMs, APIs, and Tokens — What Your Agent Thinks With

> *"To build an agent, you must first understand the mind it uses. Not deeply — you don't need to build a car engine to drive. But enough to know when it's running low on fuel."*

---

## What You'll Learn

- What Large Language Models (LLMs) are and how they work (non-technically)
- What tokens are and why they matter for your agent's budget and behavior
- How to call an LLM API directly in Python
- The difference between models: GPT-4o, Claude, Llama, Qwen
- How to choose the right model for your agent task

---

## 2.1 What Is an LLM, Really?

A Large Language Model (LLM) is a mathematical function that predicts the next word (technically: token) given everything written so far.

That's it. That's the core mechanism.

When you ask GPT-4o "What is the capital of France?", it doesn't "look it up" anywhere. It predicts, token by token, the most likely continuation of that sentence based on patterns learned from trillions of words of text.

```
Input:  "What is the capital of France?"
Token 1: "Paris"
Token 2: "."  
Done.
```

When you ask it to write code, it predicts the most likely Python code that would follow. When you ask it to plan tasks, it predicts the most likely plan.

> 📝 **SIDE NOTE: What does "trained on trillions of words" actually mean?**
>
> Training an LLM means showing it billions of examples of text from the internet, books, code, and scientific papers, and adjusting its internal mathematical parameters to be better at predicting what comes next. GPT-4o was trained on roughly 1 trillion tokens of text. Anthropic's Claude was trained on similar scales. This "knowledge" is compressed into the model's parameters — about 70-200 billion numbers for large models.

---

## 2.2 What Are Tokens?

Tokens are the "atoms" of text for LLMs. They're not exactly words, not exactly characters. They're chunks:

```
"Hello world" = 2 tokens    ["Hello", " world"]
"Tokenization" = 3 tokens   ["Token", "ization"]
"AI" = 1 token              ["AI"]
"deepagents" = 2-3 tokens   ["deep", "agents"] or ["deep", "agent", "s"]
```

A rough rule: **1 token ≈ 0.75 words** in English.

**Why tokens matter for building agents:**

1. **Context Window**: Every LLM has a maximum number of tokens it can "see" at once. GPT-4o: 128,000 tokens. Claude 3.7: 200,000 tokens. This is why filesystem middleware in DeepAgents is so important — it prevents your agent from running out of context by offloading data to files.

2. **Cost**: APIs charge per token. GPT-4o costs $2.50 per million input tokens. A long agent run might use 50,000-200,000 tokens. Know your token budget.

3. **Speed**: More tokens = slower response. Local models especially feel this.

```python
# Let's count tokens ourselves
# Install: pip install tiktoken
import tiktoken

# Get the tokenizer for GPT-4o
enc = tiktoken.encoding_for_model("gpt-4o")

text = "The quick brown fox jumps over the lazy dog"
tokens = enc.encode(text)

print(f"Text: '{text}'")           # The text
print(f"Token count: {len(tokens)}")  # How many tokens
print(f"Tokens: {tokens}")           # The raw token numbers
print(f"Token strings: {[enc.decode([t]) for t in tokens]}")  # Human readable
```

Output:
```
Text: 'The quick brown fox jumps over the lazy dog'
Token count: 9
Tokens: [791, 4062, 14198, 39935, 35308, 927, 279, 16053, 5679]
Token strings: ['The', ' quick', ' brown', ' fox', ' jumps', ' over', ' the', ' lazy', ' dog']
```

> 📝 **SIDE NOTE: What is a Context Window?**
>
> Imagine you're reading a very long book but can only hold 10 pages in your memory at once. As you read forward, the earlier pages fade. LLMs work similarly: they can only "remember" the last N tokens of a conversation. For agents, this is critical: if your agent processes a 50-page document, it might overflow its context window. DeepAgents' filesystem middleware solves this by letting the agent write results to files and read them back selectively, instead of keeping everything in memory.

---

## 2.3 Calling an LLM Directly with Python

Before using DeepAgents, let's understand the raw API call:

```python
# direct_llm_call.py
# The most basic way to call an LLM

import os
from dotenv import load_dotenv

# Load API keys from .env
load_dotenv()

# --- Option A: OpenAI ---
from openai import OpenAI

client = OpenAI()  # Automatically uses OPENAI_API_KEY env variable

response = client.chat.completions.create(
    # The model to use
    model="gpt-4o",
    
    # The messages - a list of turn-by-turn conversation
    messages=[
        {
            "role": "system",     # The hidden instructions
            "content": "You are a helpful assistant who answers in exactly one sentence."
        },
        {
            "role": "user",       # The user's message
            "content": "What is a neural network?"
        }
    ],
    
    # How random/creative the response is (0=deterministic, 1=creative)
    temperature=0.7,
    
    # Maximum tokens in the response
    max_tokens=100
)

# Extract the text from the response
text = response.choices[0].message.content
print(f"Response: {text}")

# See token usage
print(f"Input tokens: {response.usage.prompt_tokens}")
print(f"Output tokens: {response.usage.completion_tokens}")
print(f"Total tokens: {response.usage.total_tokens}")
```

```python
# --- Option B: Anthropic (Claude) ---
import anthropic

client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY

message = client.messages.create(
    model="claude-sonnet-4-5-20250929",  # Current Claude model
    max_tokens=100,
    system="You are a helpful assistant who answers in exactly one sentence.",
    messages=[
        {"role": "user", "content": "What is a neural network?"}
    ]
)

print(message.content[0].text)
```

```python
# --- Option C: Ollama (Local, Free) ---
from langchain_ollama import OllamaLLM

# Create a connection to your local Ollama server
llm = OllamaLLM(
    model="qwen2.5:7b",           # The model you pulled with 'ollama pull'
    base_url="http://localhost:11434"  # Ollama's local address
)

# Call it directly
response = llm.invoke("What is a neural network? Answer in one sentence.")
print(response)
```

> 📝 **SIDE NOTE: What is `temperature`?**
>
> Temperature controls how "random" the LLM's output is. At temperature=0, the model always picks the most likely next token — completely deterministic. At temperature=1, it samples from the distribution of likely tokens, adding creativity and variety. For agents doing research or coding, lower temperatures (0.1-0.3) work well. For creative writing agents, higher temperatures (0.7-1.0) are better. DeepAgents defaults to around 0.5.

---

## 2.4 How LangChain Unifies Model Access

Notice that OpenAI, Anthropic, and Ollama all have different APIs. LangChain solves this by providing a **unified interface**:

```python
# langchain_models.py
# LangChain lets you swap models without changing your agent code

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama

# All three have the EXACT SAME interface
gpt4 = ChatOpenAI(model="gpt-4o", temperature=0.7)
claude = ChatAnthropic(model="claude-sonnet-4-5-20250929")
qwen = ChatOllama(model="qwen2.5:7b")

# Same call for all three!
from langchain_core.messages import HumanMessage

message = HumanMessage(content="What is 2 + 2? Answer with just the number.")

print("GPT-4o says:", gpt4.invoke([message]).content)
print("Claude says:", claude.invoke([message]).content)
print("Qwen says:", qwen.invoke([message]).content)
```

This is why DeepAgents accepts model names like `"openai:gpt-4o"` or `"anthropic:claude-sonnet-4-5-20250929"` or `"ollama:qwen2.5:7b"` — it uses LangChain's unified connectors under the hood.[cite:18]

---

## 2.5 Choosing the Right Model

Here's a practical guide based on real production experience:

| Model | Best For | Context | Cost | Speed |
|-------|----------|---------|------|-------|
| **GPT-4o** | General agents, coding | 128K | ~$2.50/M tokens | Fast |
| **Claude 3.7 Sonnet** | Long research, writing | 200K | ~$3/M tokens | Fast |
| **GPT-4o-mini** | Simple tasks, testing | 128K | ~$0.15/M tokens | Very fast |
| **Qwen2.5:14b (Ollama)** | Free local agents | 32K | Free | Medium |
| **Llama3.1:8b (Ollama)** | Quick local testing | 128K | Free | Fast locally |
| **Qwen2.5:7b (Ollama)** | Low-RAM systems | 32K | Free | Fast locally |

**Rule of thumb from building production agents:**
- Use **Claude** for agents that need to follow complex multi-step instructions reliably
- Use **GPT-4o** for coding agents (excellent code generation)
- Use **Qwen2.5:14b** for free local use when you need good reasoning
- Use **GPT-4o-mini** for simple tools and rapid prototyping

> 📝 **SIDE NOTE: What does "14b" mean in model names?**
>
> "b" stands for billion parameters. More parameters = generally smarter but slower and requires more memory. Here's a rough hardware guide:
> - **7B model**: Needs ~8GB RAM (or GPU VRAM)
> - **14B model**: Needs ~16GB RAM
> - **32B model**: Needs ~32GB RAM
> - **70B model**: Needs ~48GB RAM (or multiple GPUs)
>
> For most laptop users, 7B or 14B models are the sweet spot.

---

## 2.6 The Message Format That Powers Everything

Every LLM interaction, under the hood, is a list of messages. Understanding this is essential for building agents:

```python
# message_types.py
# Understanding the 4 message types in LangChain

from langchain_core.messages import (
    SystemMessage,    # The "instructions" message (from developer)
    HumanMessage,     # Message from the user
    AIMessage,        # Message from the AI/LLM
    ToolMessage       # Result of a tool call
)

# A typical agent conversation looks like this:
conversation = [
    SystemMessage(content="You are a research assistant. Use tools to answer questions."),
    HumanMessage(content="What is the population of India?"),
    AIMessage(
        content="",  # Empty - the AI decides to use a tool instead of answering directly
        tool_calls=[{
            "name": "internet_search",
            "args": {"query": "India population 2025"},
            "id": "call_123"
        }]
    ),
    ToolMessage(
        content="India's population is approximately 1.45 billion as of 2025",
        tool_call_id="call_123"  # Links back to the AI's tool call
    ),
    AIMessage(
        content="India's population is approximately 1.45 billion as of 2025."
    )
]

# This is exactly the list of messages DeepAgents maintains internally
# and passes to the LLM on each iteration of the agent loop
```

This message list is **the agent's memory** for the current session. Each time the agent calls the LLM, it sends this entire history. When it calls a tool, the result gets added as a `ToolMessage`. The LLM then reads all previous messages to decide what to do next.

---

## 2.7 What the DeepAgents Model String Format Means

When you write:
```python
agent = create_deep_agent(model="openai:gpt-4o")
```

The string `"openai:gpt-4o"` is parsed as `provider:model_name`:

```python
# deepagents internally does something like:
provider, model_name = "openai:gpt-4o".split(":", 1)
# provider = "openai"
# model_name = "gpt-4o"

# Then it creates the appropriate LangChain model:
if provider == "openai":
    llm = ChatOpenAI(model=model_name)
elif provider == "anthropic":
    llm = ChatAnthropic(model=model_name)
elif provider == "ollama":
    llm = ChatOllama(model=model_name)
```

This is why swapping models is as simple as changing the string.[cite:32]

---

## Summary

- LLMs predict the next token — that's their core mechanism
- Tokens are word-chunks; count them to manage context and costs
- LangChain provides a unified interface across all model providers
- Choose your model based on task complexity, context needs, and budget
- The message history (SystemMessage, HumanMessage, AIMessage, ToolMessage) is the agent's working memory

**Next chapter:** We build our first real DeepAgent.

---

## Practice Questions

1. If a document is 10,000 words, approximately how many tokens is it?
2. What happens to the early messages in a conversation when you exceed the context window limit?
3. What is the difference between `temperature=0` and `temperature=1`?
4. If you're building a coding agent and cost doesn't matter, which model would you choose based on this chapter?

**Answers in Appendix B.**

---

[← Chapter 1](../chapter-01/chapter-01.md) | [Chapter 3 →](../chapter-03/chapter-03.md)
