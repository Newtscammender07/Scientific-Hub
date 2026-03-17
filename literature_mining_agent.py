from crewai import Agent
from src.tools.arxiv_search import search_arxiv

class LiteratureMiningAgent:
    def __init__(self):
        pass

    def get_agent(self, llm=None, config=None):
        config = config or {}
        is_turbo = config.get('turbo', False)
        
        return Agent(
            role='Literature Mining Specialist',
            goal=f"{'BE CONCISE: ' if is_turbo else ''}Identify and synthesize the most relevant and recent research papers for a given topic.",
            backstory="""You are an expert academic researcher with a knack for finding high-impact papers. 
            You can quickly scan ArXiv and Semantic Scholar to identify key developments, 
            pioneering methods, and foundational theories in any scientific field.""",
            llm=llm,
            verbose=False,
            memory=config.get('memory', True),
            max_iter=config.get('max_iter', 10),
            allow_delegation=False
        )
