import numpy as np
from sentence_transformers import SentenceTransformer
from crewai.tools import tool

# Global model instance
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

@tool("compute_novelty_score")
def compute_novelty_score(proposal_summary: str, existing_paper_summaries: list[str]):
    """
    Compute a novelty score (0 to 1) for a proposal against a set of existing papers.
    A higher score means higher novelty (lower similarity).
    """
    if not existing_paper_summaries:
        return 1.0
        
    model = get_model()
    proposal_embedding = model.encode([proposal_summary])[0]
    existing_embeddings = model.encode(existing_paper_summaries)
    
    # Calculate cosine similarity
    similarities = np.dot(existing_embeddings, proposal_embedding) / (
        np.linalg.norm(existing_embeddings, axis=1) * np.linalg.norm(proposal_embedding)
    )
    
    max_similarity = np.max(similarities)
    novelty_score = 1.0 - max_similarity
    return float(novelty_score)

@tool("generate_proposal_outline")
def generate_proposal_outline(gap_description: str, agency: str = "General"):
    """
    Generate a structured proposal outline based on a research gap.
    Includes common sections like Problem Statement, Methodology, and Impact.
    """
    templates = {
        "IEEE": ["Abstract", "Introduction", "Related Work", "Proposed Methodology", "Experimental Setup", "Conclusion"],
        "NSF": ["Project Summary", "Project Description", "Broader Impacts", "Intellectual Merit", "References Cited"],
        "General": ["Title", "Problem Statement", "Proposed Solution", "Implementation Plan", "Evaluation Metrics", "Budget Justification"]
    }
    
    sections = templates.get(agency, templates["General"])
    outline = f"### Research Proposal Outline for: {gap_description}\n\n"
    for section in sections:
        outline += f"#### {section}\n[Self-generated content for {section} based on the gap: {gap_description}]\n\n"
    
    return outline
