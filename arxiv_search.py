import arxiv
from crewai.tools import tool

@tool("search_arxiv")
def search_arxiv(query: str, max_results: int = 10):
    """
    Search for research papers on ArXiv.
    Returns a list of dictionaries containing title, summary, authors, and link.
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    results = []
    try:
        for result in client.results(search):
            results.append({
                "title": result.title,
                "summary": result.summary,
                "authors": [author.name for author in result.authors],
                "published": result.published.strftime("%Y-%m-%d"),
                "pdf_url": result.pdf_url,
                "entry_id": result.entry_id
            })
    except Exception as e:
        return f"Error searching ArXiv: {str(e)}"
    return results
