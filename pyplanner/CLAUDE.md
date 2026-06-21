# pyplanner

Pluggable LLM task planner for household robotics (AI2-THOR). Given a task description + visible objects, generates a step-by-step action plan.

## Usage

```python
import pyplanner

# Via registry (recommended)
planner = pyplanner.get("ReAct", host="http://ollama.aistations.org", model="llama3.2")
steps, metrics = planner.generate_plan(
    task="Make a cup of coffee using the coffee machine",
    obs="Kitchen scene loaded.",
    visible_objects=["CoffeeMachine", "Mug", "CounterTop"],
)

# Via shorthand
planner = pyplanner.react(model="llama3.2", max_steps=15)
```

## Registry keys

| Key | Class | Strategy |
|-----|-------|----------|
| `"Direct"` | `DirectPlanner` | Single LLM call → JSON plan |
| `"CoT"` | `CoTPlanner` | Reason first, then plan (2 calls) |
| `"Few-Shot CoT"` | `FewShotPlanner` | Built-in examples anchor output |
| `"Self-Refine"` | `SelfRefinePlanner` | Generate → critique → fix loop |
| `"ReAct"` | `ReActPlanner` | One Thought+Action per call (N calls, most accurate) |
| `"Hierarchical"` | `HierarchicalPlanner` | Sub-goals → action expansion |
| `"LLM Router"` | `LLMRouterPlanner` | Local generate + external API verify |
| `"Hierarchical Few-Shot"` | `HierarchicalFewShotPlanner` | Hierarchical + dynamic example retrieval |

## Output

```python
steps: list[dict]      # [{"action": "MoveTo", "object": "Kitchen", ...}, ...]
metrics: PlanMetrics   # .latency_s, .llm_calls, .input_tokens, .output_tokens, .parse_ok
```

Step fields: `action`, `object` (CamelCase AI2-THOR name), `target` (for Place), `reason`.

Valid actions: `MoveTo Find Pick Place PutIn Open Close TurnOn TurnOff Wash Sit LieOn Serve Wait`

## Backends

Configured per-planner via constructor args:
```python
planner = pyplanner.get("ReAct",
    host="http://ollama.aistations.org",   # Ollama URL
    model="llama3.2",
    provider="ollama",                      # "ollama" | "openai" | "gemini"
    api_key="",                             # for openai/gemini
)
```

## Key files

| File | What it does |
|------|-------------|
| `pyplanner/base.py` | `BasePlanner` ABC, `PlanMetrics`, `LLMBackend` (handles Ollama/OpenAI/Gemini) |
| `pyplanner/__init__.py` | `REGISTRY`, `get()`, shorthand factories |
| `pyplanner/react.py` | ReAct implementation (most used) |
| `pyplanner/hierarchical.py` | Hierarchical planner |
| `apps/thor_knowledge.py` | Task/scene definitions for demo apps |
| `apps/app.py` | Standalone Streamlit demo |
