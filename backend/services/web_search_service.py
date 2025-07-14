import os
import httpx
from typing import Dict, Any, List
from dotenv import load_dotenv

load_dotenv()

class WebSearchService:
    def __init__(self):
        self.serpapi_key = os.getenv("SERPAPI_KEY")
    
    async def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """Perform web search using SerpAPI"""
        
        if not self.serpapi_key:
            return [{"error": "SerpAPI key not configured"}]
        
        try:
            async with httpx.AsyncClient() as client:
                params = {
                    "q": query,
                    "api_key": self.serpapi_key,
                    "engine": "google",
                    "num": num_results
                }
                
                response = await client.get("https://serpapi.com/search", params=params)
                data = response.json()
                
                # Extract organic results
                organic_results = data.get("organic_results", [])
                
                results = []
                for result in organic_results:
                    results.append({
                        "title": result.get("title", ""),
                        "snippet": result.get("snippet", ""),
                        "link": result.get("link", "")
                    })
                
                return results
        
        except Exception as e:
            return [{"error": f"Search failed: {str(e)}"}]
    
    async def get_search_context(self, query: str) -> str:
        """Get search results formatted as context"""
        
        results = await self.search(query)
        
        if not results or (len(results) == 1 and "error" in results[0]):
            return "No web search results available."
        
        context_parts = []
        for i, result in enumerate(results[:3], 1):  # Use top 3 results
            if "error" not in result:
                context_parts.append(f"{i}. {result['title']}\n{result['snippet']}")
        
        return "\n\n".join(context_parts)