from crewai import Agent
from src.tools.proposal_tools import compute_novelty_score, generate_proposal_outline

class MethodologyDesignAgent:
    def __init__(self):
        self.tools = []

    def get_agent(self, llm=None, config=None):
        config = config or {}
        is_turbo = config.get('turbo', False)
        
        return Agent(
            role='Experimental Design Specialist',
            goal=f"{'BE CONCISE: ' if is_turbo else ''}Propose robust methodologies, datasets, and evaluation metrics for a given research gap.",
            backstory="""You are a methodical scientist with a deep understanding of experimental rigor. 
            You know how to design studies that are both ambitious and feasible, 
            selecting the best baselines and metrics to prove a hypothesis.""",
            tools=self.tools,
            llm=llm,
            verbose=False,
            memory=config.get('memory', True),
            max_iter=config.get('max_iter', 10),
            allow_delegation=False
        )

class GrantWritingAgent:
    def __init__(self):
        pass

    def get_agent(self, llm=None, config=None):
        config = config or {}
        is_turbo = config.get('turbo', False)
        
        return Agent(
            role='Grant Proposal Writer',
            goal=f"{'BE CONCISE: ' if is_turbo else ''}Generate structured, high-quality grant proposals and research papers.",
            backstory="""You are a professional academic writer who has secured millions in funding. 
            You know how to frame research problems to appeal to reviewers and how to 
            structure a proposal according to strict agency guidelines.""",
            tools=[generate_proposal_outline],
            llm=llm,
            verbose=False,
            memory=config.get('memory', True),
            max_iter=config.get('max_iter', 10),
            allow_delegation=True
        )

class PlagiarismNoveltyAgent:
    def __init__(self):
        pass

    def get_agent(self, llm=None, config=None):
        config = config or {}
        is_turbo = config.get('turbo', False)
        
        return Agent(
            role='Novelty & Plagiarism Auditor',
            goal=f"{'BE CONCISE: ' if is_turbo else ''}Evaluate the novelty of a proposal and ensure it does not overlap with existing work.",
            backstory="""You are a critical reviewer with an encyclopedic knowledge of prior work. 
            You use semantic similarity tools to ensure that every proposal is truly original 
            and provides a clear advancement over the state-of-the-art.""",
            tools=[compute_novelty_score],
            llm=llm,
            verbose=False,
            memory=config.get('memory', True),
            max_iter=config.get('max_iter', 10),
            allow_delegation=False
        )
