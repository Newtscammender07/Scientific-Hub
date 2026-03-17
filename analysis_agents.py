from crewai import Agent
from src.tools.semantic_analyzer import analyze_trends, identify_gaps

class TrendAnalysisAgent:
    def __init__(self):
        pass

    def get_agent(self, llm=None, config=None):
        config = config or {}
        is_turbo = config.get('turbo', False)
        
        return Agent(
            role='Trend Analyst',
            goal=f"{'BE CONCISE: ' if is_turbo else ''}Analyze a collection of research papers to identify temporal trends and emerging themes.",
            backstory="""You are a data-driven researcher with expertise in scientometrics. 
            You can see the big picture and identify how a field is moving based on publication data.""",
            llm=llm,
            verbose=False,
            memory=config.get('memory', True),
            max_iter=config.get('max_iter', 10),
            allow_delegation=False
        )

class GapIdentificationAgent:
    def __init__(self):
        pass

    def get_agent(self, llm=None, config=None):
        config = config or {}
        is_turbo = config.get('turbo', False)
        
        return Agent(
            role='Research Gap Strategist',
            goal=f"{'BE CONCISE: ' if is_turbo else ''}Critically evaluate existing literature to identify significant unexplored areas or research gaps.",
            backstory="""You have a critical eye for what is missing in a research field. 
            You can identify the 'white spaces' where new research can have the most impact.""",
            llm=llm,
            verbose=False,
            memory=config.get('memory', True),
            max_iter=config.get('max_iter', 10),
            allow_delegation=False
        )
