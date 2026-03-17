"""
Diagnostic script: fires a real CrewAI mini-run with a step callback
and prints EVERYTHING about the step object so we know the exact structure.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

captured_steps = []
captured_tasks = []

def step_callback(step):
    info = {
        "type": type(step).__name__,
        "module": type(step).__module__,
        "attrs": {k: str(getattr(step, k, None))[:120] for k in dir(step) 
                  if not k.startswith('_') and not callable(getattr(step, k, None))}
    }
    captured_steps.append(info)
    print(f"\n>>> STEP CALLBACK FIRED <<<")
    print(f"  Type: {info['type']} from {info['module']}")
    for k, v in info['attrs'].items():
        if v and v != 'None':
            print(f"  .{k} = {v[:80]}")

def task_callback(task):
    info = {
        "type": type(task).__name__,
        "attrs": {k: str(getattr(task, k, None))[:80] for k in ['description', 'output', 'agent'] 
                  if hasattr(task, k)}
    }
    captured_tasks.append(info)
    print(f"\n>>> TASK CALLBACK FIRED <<<")
    for k, v in info['attrs'].items():
        print(f"  .{k} = {v}")

if __name__ == "__main__":
    from src.crew import ResearchCrew
    
    print("Starting mini test run with topic: 'AI'")
    print("(turbo_mode=True for speed, max_iter=2)")
    
    crew = ResearchCrew(
        topic="AI in healthcare",
        model_name="llama3.2:latest",
        provider="Ollama",
        task_callback=task_callback,
        step_callback=step_callback,
        turbo_mode=True
    )
    
    try:
        result = crew.kickoff()
        print(f"\n=== RUN COMPLETE ===")
        print(f"Step callbacks received: {len(captured_steps)}")
        print(f"Task callbacks received: {len(captured_tasks)}")
        if not captured_steps:
            print("WARNING: No step callbacks were fired at all!")
        if not captured_tasks:
            print("WARNING: No task callbacks were fired at all!")
    except Exception as e:
        print(f"\n=== ERROR: {e} ===")
        print(f"Step callbacks received before error: {len(captured_steps)}")
        print(f"Task callbacks received before error: {len(captured_tasks)}")
