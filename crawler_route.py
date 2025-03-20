# Language: Python
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from urllib.parse import urlparse

from extensions.crawler import (
    crawl_parallel,
    get_urls_from_xml,
    extract_links_from_url  # Add this import
)

load_dotenv()

# Initialize OpenAI, Supabase clients, etc.
# (Other configuration and global variables remain here)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PORT = int(os.getenv("BACKEND_PORT", 8000))
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
# ...

# Configure FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    scrape_type: str
    url: str
    supabase_table: str
    source_name: str
    expand_details: str

class SpiderRequest(BaseModel):
    url: str
    max_depth: int

@app.post("/crawl")
async def query_model(request: QueryRequest):
    scrape_type = request.scrape_type
    url = request.url
    supabase_table = request.supabase_table
    source_name = request.source_name
    expand_details = request.expand_details

    if scrape_type == "XML":
        print(f"Fetching URLs from XML sitemap: {url}")
        urls = get_urls_from_xml(url)
        print("Found URLs:")
        for idx, found_url in enumerate(urls, 1):
            print(f"{idx}. {found_url}")
        print(f"Total URLs found: {len(urls)}")
        if not urls:
            return {"message": "No URLs found in the XML sitemap."}, 404
    else:
        urls = [u.strip() for u in url.split(',')]
        print(f"Multiple URLs mode: {urls}")

    print(f"Found {len(urls)} URLs to crawl")
    await crawl_parallel(urls, supabase_table, source_name, expand_details)
    return {"message": "Crawling started successfully.", "url_count": len(urls)}, 200

@app.post("/spider")
async def spider_website(request: SpiderRequest):
    url = request.url
    max_depth = request.max_depth

    print(f"Spidering website: {url}")
    
    # Check if the URL is valid
    try:
        parsed_url = urlparse(url)
        if not all([parsed_url.scheme, parsed_url.netloc]):
            return {"message": "Invalid URL format. Please provide a valid URL with http:// or https:// prefix."}, 400
    except Exception:
        return {"message": "Invalid URL format."}, 400
    
    try:
        # Spider the website to get all links
        links = await extract_links_from_url(url, max_depth)
        
        # Filter out duplicates and sort
        unique_links = sorted(list(set(links)))
        
        return {
            "message": "Website spidered successfully",
            "url": url,
            "link_count": len(unique_links),
            "links": unique_links
        }
    except Exception as e:
        print(f"Error spidering website: {str(e)}")
        return {"message": f"Error spidering website: {str(e)}"}, 500

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)