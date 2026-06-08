# Appendix B: Practice Question Answers & Troubleshooting

## Chapter 1 Answers

1. `source venv/bin/activate` activates the virtual environment, making all pip installs go to that isolated folder instead of system Python.
2. API keys in code files can be accidentally committed to Git and exposed publicly, where bots steal them within minutes.
3. Ollama runs LLMs locally on your computer. OpenAI API sends your data to OpenAI's servers and charges per token.
4. `os.getenv("OPENAI_API_KEY", "")` returns an empty string `""` if the key is not set (the second argument is the default).

## Chapter 2 Answers

1. 10,000 words × (4/3 tokens per word) ≈ 13,333 tokens.
2. The LLM truncates or summarizes older messages, or raises a context length error.
3. `temperature=0`: Always picks the most likely token (deterministic). `temperature=1`: Randomly samples from the distribution (creative/random).
4. GPT-4o (excellent code generation).

## Chapter 3 Answers

1. Built-in tools: `write_todos`, `ls`, `read_file`, `write_file`, `edit_file`, `task` (subagent delegation).
2. `invoke()` waits for the full result. `stream()` yields intermediate updates as they happen.
3. No, without a checkpointer each `invoke()` starts a fresh session with no memory of previous calls.
4. `-1` as a list index returns the LAST item in the list.

## Chapter 4 Answers

1. The filesystem lets the agent write large results to files and read only what it needs, instead of keeping everything in the limited context window.
2. Each subagent runs in an isolated context, returning only a clean summary to the main agent. This prevents the main agent's context from being overwhelmed.
3. `gpt-4o-mini` is cheaper. Math doesn't need the most powerful model.
4. The `@tool` decorator transforms a Python function into a LangChain Tool object by reading the function's name, docstring, and type hints to create a description the LLM can understand.

---

## Common Errors and Fixes

### Error: `ModuleNotFoundError: No module named 'deepagents'`
```bash
# Fix: Install deepagents in your virtual environment
pip install deepagents
# Make sure your virtual environment is activated first!
```

### Error: `AuthenticationError: Incorrect API key`
```bash
# Fix: Check your .env file has the correct key
cat .env  # Should show OPENAI_API_KEY=sk-...
# Make sure load_dotenv() is called at the top of your script
```

### Error: `Connection refused: http://localhost:11434`
```bash
# Fix: Start Ollama first
ollama serve
# In a separate terminal, then run your Python script
```

### Error: `Context length exceeded`
```bash
# Fix: The agent's conversation is too long.
# Solutions:
# 1. Use a model with larger context (Claude 3.7: 200K tokens)
# 2. Start a new thread (new thread_id)
# 3. Enable filesystem middleware so agent offloads data to files
```

### Error: Agent does nothing / doesn't call tools
```bash
# Fix: Improve your tool docstrings
# The LLM reads docstrings to know when to use tools.
# Add specific examples and conditions to your docstrings.
```

### Agent loops forever
```bash
# Fix: Add a recursion limit
config = {
    "configurable": {"thread_id": "my-session"},
    "recursion_limit": 50  # Default is 25, max useful is ~100
}
```

---

## Recommended Next Resources

- **LangChain Docs**: https://docs.langchain.com/oss/python/deepagents
- **LangChain Academy** (Free course): https://academy.langchain.com
- **DeepAgents GitHub**: https://github.com/langchain-ai/deepagents
- **LangGraph Docs**: https://langchain-ai.github.io/langgraph/
- **LangSmith** (Observability): https://smith.langchain.com
- **Ollama**: https://ollama.ai
- **Tavily** (Web search API): https://tavily.com
