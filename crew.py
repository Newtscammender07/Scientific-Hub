from crewai import Crew, Task, Process, LLM
from src.agents.literature_mining_agent import LiteratureMiningAgent
from src.agents.analysis_agents import TrendAnalysisAgent, GapIdentificationAgent
from src.agents.writing_agents import MethodologyDesignAgent, GrantWritingAgent, PlagiarismNoveltyAgent
import os
import sys
import contextlib

@contextlib.contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    # Force TERM=dumb to prevent rich from using legacy windows render
    os.environ["TERM"] = "dumb"
    
    with open(os.devnull, 'w') as fnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = fnull
        sys.stderr = fnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

class ResearchCrew:
    def __init__(self, topic: str, model_name: str = "gpt-4o-mini", provider: str = "OpenAI", 
                 task_callback=None, step_callback=None, turbo_mode: bool = False, flash_mode: bool = False):
        self.topic = topic
        self.task_callback = task_callback
        self.step_callback = step_callback
        self.turbo_mode = turbo_mode
        self.flash_mode = flash_mode
        self.provider = provider

        
        # Performance configuration
        if flash_mode:
            self.agent_config = {"max_iter": 1, "memory": False, "turbo": True}
            mode_label = "FLASH"
        elif turbo_mode:
            self.agent_config = {"max_iter": 3, "memory": False, "turbo": True}
            mode_label = "TURBO"
        else:
            self.agent_config = {"max_iter": 10, "memory": True, "turbo": False}
            mode_label = "Standard"


        # Configure LLM
        if provider == "OpenAI":
            self.llm = LLM(model=f"openai/{model_name}", api_key=os.getenv("OPENAI_API_KEY"), max_retries=3)
        elif provider == "Gemini":
            key = os.getenv("GEMINI_API_KEY")
            if not key:
                print("DEBUG: GEMINI_API_KEY not found in environment!")
            else:
                key_str = str(key)
                print(f"DEBUG: Using Gemini Key ending in ...{key_str[-4:] if len(key_str) > 4 else key_str}")
            self.llm = LLM(model=f"gemini/{model_name}", api_key=key, max_retries=5)
        elif provider == "Groq":
            self.llm = LLM(
                model=f"groq/{model_name}",
                api_key=os.getenv("GROQ_API_KEY"),
                max_retries=3,
                max_tokens=1200
            )
        else:
            # Ultra Fix v6.0: Force litellm to use the OpenAI-bridge handler for Ollama.
            clean_model_name = model_name.strip()
            print(f">>>>> INITIALIZING OLLAMA ULTRA-FIX v6 | Model: '{clean_model_name}' | Provider: OpenAI Bridge <<<<<")
            
            self.llm = LLM(
                model=f"openai/{clean_model_name}", 
                base_url="http://127.0.0.1:11434/v1",
                api_key="ollama",
                timeout=120
            ) 

        # Initialize agents with optimized config
        self.lit_agent     = LiteratureMiningAgent().get_agent(self.llm, config=self.agent_config)
        self.trend_agent   = TrendAnalysisAgent().get_agent(self.llm, config=self.agent_config)
        self.gap_agent     = GapIdentificationAgent().get_agent(self.llm, config=self.agent_config)
        self.method_agent  = MethodologyDesignAgent().get_agent(self.llm, config=self.agent_config)
        self.writing_agent = GrantWritingAgent().get_agent(self.llm, config=self.agent_config)
        self.novelty_agent = PlagiarismNoveltyAgent().get_agent(self.llm, config=self.agent_config)

        # Groq does not support OpenAI-style function calling → strip tools
        # so the model never receives a function schema and avoids tool_use_failed errors.
        if self.provider == "Groq":
            self.writing_agent.tools = []
            self.novelty_agent.tools = []
            # Disable delegation for Groq to avoid tool_use_failed errors
            for agent in [self.lit_agent, self.trend_agent, self.gap_agent, 
                         self.method_agent, self.writing_agent, self.novelty_agent]:
                agent.allow_delegation = False


    def setup_flash_crew(self):
        """2-task express pipeline. Targets <2 min on a local Ollama model."""
        from crewai import Agent
        
        EXPRESS_LIMIT = "CRITICAL: Reply in PLAIN TEXT ONLY. Max 120 words total. No preamble or filler."
        
        analyst = Agent(
            role="Express Research Analyst",
            goal=f"Rapidly identify key research gaps and trends for the topic: {self.topic}",
            backstory="You are an ultra-efficient analyst who gives sharp, dense, bullet-point answers.",
            llm=self.llm, verbose=False, memory=False, max_iter=1, allow_delegation=False
        )
        
        writer = Agent(
            role="Express Grant Writer",
            goal="Produce a short, well-structured grant proposal skeleton.",
            backstory="You write ultra-concise grant outlines. Every word counts. No fluff.",
            llm=self.llm, verbose=False, memory=False, max_iter=1, allow_delegation=False
        )
        
        analysis_task = Task(
            description=(
                f"{EXPRESS_LIMIT}\n\n"
                f"Topic: {self.topic}\n"
                f"Do THREE things:\n"
                f"1. Name 3 recent research themes (1 line each)\n"
                f"2. Identify the single biggest research gap (2 lines)\n"
                f"3. Suggest one concrete methodology to address it (2 lines)"
            ),
            expected_output="3 themes, 1 gap, 1 method. Max 120 words.",
            agent=analyst,
            callback=self.task_callback
        )
        
        proposal_task = Task(
            description=(
                f"{EXPRESS_LIMIT}\n\n"
                f"Write a grant proposal SKELETON for topic: {self.topic}\n"
                f"Sections: 1) Problem (2 lines) 2) Proposed Method (2 lines) 3) Expected Impact (1 line)\n"
                f"Use the analysis from the previous task."
            ),
            expected_output="A 5-7 line grant proposal skeleton.",
            agent=writer,
            context=[analysis_task],
            callback=self.task_callback
        )
        
        return Crew(
            agents=[analyst, writer],
            tasks=[analysis_task, proposal_task],
            process=Process.sequential,
            verbose=False,
            step_callback=self.step_callback
        )

    def setup_turbo_crew(self):
        """6-agent pipeline optimised for Groq — structured, actionable output."""

        lit_task = Task(
            description=(
                f"Search ArXiv and Semantic Scholar for the 7 most important papers on: {self.topic}.\n"
                "For each paper include: Title, Authors & Year, Key Contribution (2 sentences), Methodology used."
            ),
            expected_output=(
                "7 papers. Each entry: Title | Authors (Year) | Key Contribution | Methodology. "
                "Formatted as a numbered list."
            ),
            agent=self.lit_agent
        )
        trend_task = Task(
            description=(
                f"Based on the literature, identify 5 significant research trends in {self.topic}.\n"
                "For each trend: name it, explain why it matters (2-3 sentences), and cite 1-2 supporting papers."
            ),
            expected_output=(
                "5 numbered trends. Each: Trend Name | Why It Matters | Supporting Papers."
            ),
            agent=self.trend_agent,
            context=[lit_task]
        )
        gap_task = Task(
            description=(
                "Identify the 3 most critical research gaps not addressed in the literature found.\n"
                "For each gap: describe the gap clearly, why it matters, what would solving it enable, "
                "and the biggest challenge to addressing it."
            ),
            expected_output=(
                "3 numbered research gaps. Each: Gap Description | Why It Matters | "
                "What Solving It Enables | Key Challenge."
            ),
            agent=self.gap_agent,
            context=[lit_task, trend_task]
        )
        method_task = Task(
            description=(
                "Design a concrete research methodology for the most important research gap identified.\n"
                "Include: Dataset sources (be specific), Technical approach/algorithm, "
                "Evaluation metrics, Baseline comparisons, Estimated timeline (months), "
                "and potential risks."
            ),
            expected_output=(
                "Structured methodology with: Dataset | Approach | Metrics | Baselines | Timeline | Risks."
            ),
            agent=self.method_agent,
            context=[gap_task]
        )
        writing_task = Task(
            description=(
                "Write a structured grant proposal for the proposed research. Include all sections:\n"
                "1. Problem Statement (3-4 sentences describing the gap and urgency)\n"
                "2. Proposed Approach (4-5 sentences on methodology and innovation)\n"
                "3. Expected Outcomes (3 bullet points: deliverables and impact)\n"
                "4. Broader Impact (2-3 sentences on societal/scientific value)\n"
                "5. Budget Justification (2-3 sentences on resource needs)"
            ),
            expected_output=(
                "Full grant proposal with all 5 sections clearly labelled and substantive."
            ),
            agent=self.writing_agent,
            context=[method_task, gap_task]
        )
        novelty_task = Task(
            description=(
                "Audit the grant proposal for novelty and originality.\n"
                "IMPORTANT: Start your response with exactly this line: NOVELTY_SCORE: X.XX\n"
                "(where X.XX is a number between 0.00 and 1.00)\n"
                "Then provide: Justification for the score, "
                "2 most similar existing works with explanation of overlap, "
                "what specifically makes this proposal novel/different, "
                "and 2 concrete recommendations to strengthen originality."
            ),
            expected_output=(
                "Start with: NOVELTY_SCORE: X.XX\n"
                "Then use this exact Markdown structure:\n"
                "# **Novelty Audit: Score + Justification**\n"
                "# **2 Similar Works**\n"
                "# **What Makes It Novel**\n"
                "# **2 Recommendations**"
            ),
            agent=self.novelty_agent,
            context=[writing_task, lit_task]
        )
        return Crew(
            agents=[self.lit_agent, self.trend_agent, self.gap_agent,
                    self.method_agent, self.writing_agent, self.novelty_agent],
            tasks=[lit_task, trend_task, gap_task, method_task, writing_task, novelty_task],
            process=Process.sequential,
            verbose=False,
            task_callback=self.task_callback,
            step_callback=self.step_callback
        )

    def setup_crew(self):
        # 1. Literature Discovery Task
        lit_task = Task(
            description=f"Search for and summarize the top 10 most relevant research papers related to {self.topic}.",
            expected_output="A list of paper summaries and their key contributions.",
            agent=self.lit_agent
        )

        # 2. Trend Analysis Task
        trend_task = Task(
            description=f"Analyze the papers found in the literature search. Identify temporal trends and key evolving topics.",
            expected_output=f"A report on research trends in the field of {self.topic}.",
            agent=self.trend_agent,
            context=[lit_task]
        )

        # 3. Gap Identification Task
        gap_task = Task(
            description=f"Based on the trends and literature, identify 2-3 significant research gaps or under-explored areas.",
            expected_output="A list of research gaps with descriptions and potential impact.",
            agent=self.gap_agent,
            context=[lit_task, trend_task]
        )

        # 4. Methodology Design Task
        method_task = Task(
            description="For the most promising research gap identified, propose a research methodology, potential datasets, and benchmarks.",
            expected_output="A detailed experimental design and methodology proposal.",
            agent=self.method_agent,
            context=[gap_task]
        )

        # 5. Grant Proposal Generation Task
        writing_task = Task(
            description="Draft a full grant proposal based on the proposed methodology. Follow a structured outline (Problem, Method, Impact, etc.).",
            expected_output="A full, structured grant proposal in IEEE/ACM style.",
            agent=self.writing_agent,
            context=[method_task]
        )

        # 6. Novelty and Plagiarism Audit Task
        novelty_task = Task(
            description=(
                "Audit the generated proposal for novelty and potential overlap with the papers found in the initial search.\n"
                "IMPORTANT: Start your response with exactly this line: NOVELTY_SCORE: X.XX\n"
                "(where X.XX is a number between 0.00 and 1.00)\n"
                "Then provide: justification for the score, 2 similar existing works, "
                "what makes the proposal novel, and 2 recommendations to strengthen originality."
            ),
            expected_output=(
                "Start with: NOVELTY_SCORE: X.XX\n"
                "Then use this exact Markdown structure:\n"
                "# **Novelty Audit: Score + Justification**\n"
                "# **2 Similar Works**\n"
                "# **What Makes It Novel**\n"
                "# **2 Recommendations**"
            ),
            agent=self.novelty_agent,
            context=[writing_task, lit_task]
        )

        return Crew(
            agents=[
                self.lit_agent, self.trend_agent, self.gap_agent, 
                self.method_agent, self.writing_agent, self.novelty_agent
            ],
            tasks=[lit_task, trend_task, gap_task, method_task, writing_task, novelty_task],
            process=Process.sequential,
            verbose=False,
            task_callback=self.task_callback,
            step_callback=self.step_callback
        )

    def kickoff(self):
        import re, time, sys

        def _safe_log(msg):
            try:
                sys.__stderr__.write(msg + "\n")
                sys.__stderr__.flush()
            except Exception:
                pass

        def _make_crew():
            if self.flash_mode:
                return self.setup_flash_crew()
            elif self.turbo_mode:
                return self.setup_turbo_crew()
            else:
                return self.setup_crew()

        max_attempts = 5
        for attempt in range(max_attempts):
            crew = _make_crew()   # Fresh crew every attempt — avoids reusing a half-failed run
            try:
                with suppress_stdout_stderr():
                    return crew.kickoff()
            except Exception as e:
                err_str = str(e)
                is_rate_limit = (
                    "429" in err_str or
                    "rate_limit" in err_str.lower() or
                    "RateLimitError" in err_str or
                    "Too Many Requests" in err_str or
                    "Request too large" in err_str
                )
                if is_rate_limit and attempt < max_attempts - 1:
                    match = re.search(r'try again in ([\d.]+)s', err_str, re.IGNORECASE)
                    wait_time = float(match.group(1)) + 3 if match else (25 * (attempt + 1))
                    _safe_log(f"Rate limit hit. Waiting {wait_time:.1f}s before retry #{attempt + 2}/{max_attempts}...")
                    time.sleep(wait_time)
                else:
                    raise

