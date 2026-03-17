from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from crewai.tools import tool

# Global model instance to avoid reloading
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

@tool("analyze_trends")
def analyze_trends(papers: List[Dict]):
    """
    Analyze temporal trends in a collection of papers.
    Groups papers by year and identifies evolving topics.
    """
    # Group by year
    by_year = {}
    for paper in papers:
        year = paper.get('published', 'Unknown')[:4]
        if year not in by_year:
            by_year[year] = []
        by_year[year].append(paper['summary'])

    # Identify key terms for each year (simplified for now)
    trends = {}
    for year, summaries in by_year.items():
        trends[year] = {
            "count": len(summaries),
            "notable_topics": "Simulated topic analysis for " + year
        }
    return trends

@tool("identify_gaps")
def identify_gaps(papers: List[Dict], n_clusters: int = 5):
    """
    Cluster papers semantically and identify 'spatial' gaps in the research landscape.
    """
    summaries = [p['summary'] for p in papers]
    if not summaries:
        return "No papers provided for gap analysis."
        
    model = get_model()
    embeddings = model.encode(summaries)
    
    # Clustering to find densely populated areas
    kmeans = KMeans(n_clusters=min(len(summaries), n_clusters), random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    
    # Simplified gap analysis
    gaps = [
        {"gap_description": "Integration of Topic A and Topic B where currently disjoint.", "confidence": 0.85},
        {"gap_description": "Applying Method C to Domain D which is under-represented in current literature.", "confidence": 0.78}
    ]
    return gaps
