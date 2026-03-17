"""
Test callbacks without actually doing research - just see if they get called.
Uses a direct crew instantiation with a mock LLM.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from crewai import Agent, Task, Crew, Process, LLM
from langchain_core.agents import AgentAction, AgentFinish

step_calls = []
task_calls = []

def step_cb(step):
    step_calls.append(step)
    print(f"\n[STEP] type={type(step).__name__}")
    for attr in ['tool', 'tool_input', 'log', 'thought', 'text', 'return_values', 'output']:
        val = getattr(step, attr, 'N/A')
        if val and val != 'N/A':
            print(f"  .{attr} = {str(val)[:100]}")

def task_cb(task):
    task_calls.append(task)
    print(f"\n[TASK DONE] type={type(task).__name__}, description={str(getattr(task, 'description', ''))[:60]}")

llm = LLM(
    model="openai/llama3.2:latest",
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama",
    timeout=30,  # Short timeout for testing
)

agent = Agent(
    role='Tester',
    goal='BE CONCISE. Say hello in exactly 3 words.',
    backstory='You are a test agent.',
    llm=llm,
    verbose=True,
    memory=False,
    max_iter=2,
    allow_delegation=False
)

task = Task(
    description='Say hello in exactly 3 words. Do not use any tools.',
    expected_output='Exactly 3 words.',
    agent=agent,
)

crew = Crew(
    agents=[agent],
    tasks=[task],
    process=Process.sequential,
    verbose=True,
    step_callback=step_cb,
    task_callback=task_cb,
)

try:
    result = crew.kickoff()
    print(f"\n=== DONE ===")
    print(f"Result: {result}")
    print(f"Step callbacks: {len(step_calls)}")
    print(f"Task callbacks: {len(task_calls)}")
except Exception as e:
    print(f"\n=== ERROR: {e} ===")
    print(f"Step callbacks before error: {len(step_calls)}")
    print(f"Task callbacks before error: {len(task_calls)}")
