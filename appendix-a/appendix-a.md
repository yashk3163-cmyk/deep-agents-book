# Appendix A: DeepAgents Complete API Reference

## `create_deep_agent()` — Full Signature

```python
from deepagents import create_deep_agent

agent = create_deep_agent(
    model: str | BaseChatModel,          # Required. LLM to use
    tools: list = [],                     # Optional. Custom tools
    system_prompt: str = "",             # Optional. Additional system instructions
    subagents: list = [],                # Optional. Specialist subagents
    middleware: list = [],               # Optional. Custom middleware stack
    store: BaseStore | None = None,      # Optional. Cross-session persistent store
    checkpointer: BaseCheckpointSaver | None = None  # Optional. State persistence
)
```

## Model String Format

| Format | Example | Provider |
|--------|---------|----------|
| `provider:model` | `openai:gpt-4o` | OpenAI |
| `provider:model` | `anthropic:claude-sonnet-4-5-20250929` | Anthropic |
| `provider:model` | `ollama:qwen2.5:14b` | Ollama (local) |
| Model object | `ChatOpenAI(model="gpt-4o")` | Direct LangChain model |

## Built-in Middleware

| Middleware | Import | What it Adds |
|-----------|--------|-------------|
| `TodoListMiddleware` | `from langchain.agents.middleware import TodoListMiddleware` | `write_todos` planning tool |
| `FilesystemMiddleware` | `from deepagents.middleware.filesystem import FilesystemMiddleware` | `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep` |
| `SubAgentMiddleware` | `from deepagents.middleware.subagents import SubAgentMiddleware` | `task` delegation tool |

## Backends

| Backend | Import | Use Case |
|---------|--------|----------|
| `StateBackend` | `from deepagents.backends import StateBackend` | Default. Ephemeral, per-thread |
| `StoreBackend` | `from deepagents.backends import StoreBackend` | Cross-session persistent |
| `FilesystemBackend` | `from deepagents.backends import FilesystemBackend` | Real disk access |
| `CompositeBackend` | `from deepagents.backends import CompositeBackend` | Route by path prefix |

## Common Patterns

```python
# Minimal agent
agent = create_deep_agent(model="openai:gpt-4o")

# Agent with tools
agent = create_deep_agent(model="openai:gpt-4o", tools=[my_tool])

# Agent with persistence
agent = create_deep_agent(
    model="openai:gpt-4o",
    checkpointer=MemorySaver()
)

# Agent with cross-session memory
store = InMemoryStore()
agent = create_deep_agent(
    model="openai:gpt-4o",
    store=store,
    middleware=[FilesystemMiddleware(backend=lambda rt: CompositeBackend(
        default=StateBackend(rt),
        routes={"/memories/": StoreBackend(rt)}
    ))]
)

# Fully local agent
agent = create_deep_agent(
    model="ollama:qwen2.5:14b",
    tools=[my_tool],
    system_prompt="You are a local assistant."
)
```

## invoke() vs stream()

```python
# invoke() - wait for complete response
result = agent.invoke({"messages": "Your question"}, config=config)
response = result["messages"][-1].content

# stream() - get updates as they happen
for step in agent.stream({"messages": "Your question"}, config=config, stream_mode="updates"):
    # Process each step
    if "agent" in step:
        print(step["agent"]["messages"][-1].content)
```
