def evaluate_plan(plan: list[dict], task: str, obs: str, planner) -> str:
    """Ask the LLM to assess the plan: coverage of intent objects/locations + precondition correctness."""
    from pyplanner.base import STEP_SCHEMA, ROBOT_ACTIONS
    steps_str = "\n".join(
        f"  {i+1}. {s.get('action','')} {s.get('object','')}"
        for i, s in enumerate(plan)
    )
    actions_str = ", ".join(ROBOT_ACTIONS)
    prompt = (
        f"You are evaluating a robot action plan.\n\n"
        f"Task: {task}\n\n"
        f"Current robot observation:\n{obs}\n\n"
        f"Available actions: {actions_str}\n\n"
        f"Action rules:\n{STEP_SCHEMA}\n\n"
        f"Plan to evaluate:\n{steps_str}\n\n"
        "Evaluate the plan on TWO criteria:\n\n"
        "CRITERIA 1 — Intent coverage:\n"
        "  Extract every object and location mentioned in the task "
        "(e.g. 'Grab apple in kitchen, place on dining table in living room' → "
        "apple, kitchen, dining table, living room).\n"
        "  Check that EACH one appears as the 'object' field of at least one step.\n"
        "  List any that are missing.\n\n"
        "CRITERIA 2 — Precondition correctness:\n"
        "  For each step check: (a) preconditions met "
        "(Pick requires Find first; Place requires Holding+Arrived), "
        "(b) sequence is logical.\n\n"
        "CRITERIA 3 — Implicit container actions:\n"
        "  If any object in the task is inside a container (Fridge, Cabinet, Drawer, "
        "Microwave, Box, etc.), the plan MUST include Open before Find/Pick of that object "
        "and Close after Pick. Check if these implicit steps are missing.\n\n"
        "End with an overall verdict on its own line: VALID or NEEDS_REVISION."
    )
    content, _, _ = planner._chat([{"role": "user", "content": prompt}], temperature=0.3)
    return content


def refine_plan(plan: list[dict], task: str, obs: str, evaluation: str, planner) -> list[dict]:
    """Ask the LLM to produce an improved plan that covers all intent objects/locations."""
    from pyplanner.base import STEP_SCHEMA, JSON_EXAMPLE, parse_steps
    steps_str = "\n".join(
        f"  {i+1}. {s.get('action','')} {s.get('object','')}"
        for i, s in enumerate(plan)
    )
    prompt = (
        f"You are improving a robot action plan.\n\n"
        f"Task: {task}\n\n"
        f"Current robot observation:\n{obs}\n\n"
        f"Action rules:\n{STEP_SCHEMA}\n\n"
        f"Current plan:\n{steps_str}\n\n"
        f"Evaluation (issues to fix):\n{evaluation}\n\n"
        "Requirements for the revised plan:\n"
        "1. Every object AND location mentioned in the task must appear as the "
        "'object' field of at least one step "
        "(e.g. task mentions 'kitchen' → include MoveTo Kitchen; "
        "mentions 'dining table' → include MoveTo DiningTable; "
        "mentions 'apple' → include Find Apple and Pick).\n"
        "2. All preconditions must be satisfied "
        "(Find before Pick; MoveTo receptacle before Place).\n"
        "3. Implicit container actions (REQUIRED even if not in the task text): "
        "if an object is inside a container (Fridge, Cabinet, Drawer, Microwave, etc.), "
        "add Find <container> → Open before Find <object>, "
        "and Find <container> → Close after Pick.\n"
        "4. Steps must be in logical execution order.\n\n"
        f"Output ONLY a valid JSON plan in this exact format:\n{JSON_EXAMPLE}"
    )
    content, _, _ = planner._chat([{"role": "user", "content": prompt}], temperature=0.2)
    return parse_steps(content)