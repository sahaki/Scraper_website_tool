# Language: Python
import os
import json
import asyncio
import requests
import fitz  # PyMuPDF
from datetime import datetime, timezone
from urllib.parse import urlparse
from xml.etree import ElementTree
from dataclasses import dataclass
from typing import List, Dict, Any

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from openai import AsyncOpenAI
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
embed_model = os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
embed_dim = os.getenv('SUPABASE_VECTOR_DIM', 1536)

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Global cache variables
SUMMARIZER_PROMPT: str = None 
QA_PROMPT: str = None  
REFORMAT_CHUNK_PROMPT: str = None 

@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

async def reformat_text(chunk: str) -> str:
    """Reformat the chunk text using the system prompt from reformat_chunk.md."""
    global REFORMAT_CHUNK_PROMPT
    if REFORMAT_CHUNK_PROMPT is None:
        try:
            with open('./system_prompts/reformat_chunk.md', 'r') as file:
                REFORMAT_CHUNK_PROMPT = file.read()
        except Exception as e:
            print(f"Error loading reformat prompt: {e}")
            return ''
    
    try:
        response = await openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": REFORMAT_CHUNK_PROMPT},
                {"role": "user", "content": f"Reformat the following content:\n{chunk}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in reformat_text: {e}")
        return ''

async def get_questions_and_answers(chunk: str) -> Dict[str, Any]:
    """Extract Q&A from the chunk for RAG."""
    global QA_PROMPT
    if QA_PROMPT is None:
        try:
            with open('./system_prompts/qa.md', 'r') as file:
                QA_PROMPT = file.read()
        except Exception as e:
            print(f"Error loading Q&A prompt: {e}")
            return {"qa": []}

    try:
        response = await openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": QA_PROMPT},
                {"role": "user", "content": f"Think of a topic for this content and create a question based on that topic. Chunk content:\n{chunk}..."}
            ],
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error getting Q&A: {e}")
        return {"qa": []}

async def get_summarize(chunk: str) -> str:
    """Extract a summary from the chunk for RAG."""
    global SUMMARIZER_PROMPT
    if SUMMARIZER_PROMPT is None:
        try:
            with open('./system_prompts/summarizer.md', 'r') as file:
                SUMMARIZER_PROMPT = file.read()
        except Exception as e:
            print(f"Error loading summarizer prompt: {e}")
            return ''
    try:
        response = await openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SUMMARIZER_PROMPT},
                {"role": "user", "content": f"Summarize chunk content:\n{chunk}..."}
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error getting summary: {e}")
        return ''

async def chunk_text(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # If we're at the end of the text, just take what's left
        if end >= text_length:
            content = text[start:].strip()
            if content:  # Check that the text is not empty
                chunk = await reformat_text(content)
                if chunk:
                    chunks.append(chunk)
            break

        # Try to find a code block boundary first (```)
        chunk = text[start:end]
        code_block = chunk.rfind('```')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block

        # If no code block, try to break at a paragraph
        elif '\n\n' in chunk:
            # Find the last paragraph break
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_break

        # If no paragraph break, try to break at a sentence
        elif '. ' in chunk:
            # Find the last sentence break
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_period + 1

        # Extract and verify the chunk text is not empty
        content = text[start:end].strip()
        if not content:  # If empty, skip reformat processing
            start = max(start + 1, end)
            continue

        # Extract chunk and clean it up
        chunk = await reformat_text(content)
        if chunk:
            chunks.append(chunk)

        # Move start position for next chunk
        start = max(start + 1, end)

    return chunks

async def chunk_text_with_qa(text: str, chunk_size: int = 5000) -> List[str]:
    """"Split text into chunks, respecting code blocks and paragraphs. + append Q&A data for each chunk."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # If we're at the end of the text, just take what's left
        if end >= text_length:
            content = text[start:].strip()
            if content:  # Check that the text is not empty
                chunk = await reformat_text(content)
                if chunk:
                    chunks.append(chunk)

                    # Get Summarize for this chunk and append to the list if available
                    summarize = await get_summarize(chunk)
                    if summarize:
                        chunks.append(summarize)

                    # Get Q&A for this chunk and append to the list if available
                    qa = await get_questions_and_answers(chunk)
                    if qa:
                        qa_text = ""
                        for q in qa['qa']:
                            qa_text += q.get('title', '') + '\n' + q.get('q_and_a', '') + '\n'
                        if qa_text:
                            chunks.append(qa_text)
            break

        # Try to find a code block boundary first (```)
        chunk = text[start:end]
        code_block = chunk.rfind('```')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block

        # If no code block, try to break at a paragraph
        elif '\n\n' in chunk:
            # Find the last paragraph break
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_break

        # If no paragraph break, try to break at a sentence
        elif '. ' in chunk:
            # Find the last sentence break
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_period + 1

       # Extract and verify the chunk text is not empty
        content = text[start:end].strip()
        if not content:  # If empty, skip reformat processing
            start = max(start + 1, end)
            continue

        # Extract chunk and clean it up
        chunk = await reformat_text(content)
        if chunk:
            chunks.append(chunk)

            # Get Summarize for this chunk and append to the list if available
            summarize = await get_summarize(chunk)
            if summarize:
                chunks.append(summarize)

            # Get Q&A for this chunk and append to the list if available
            qa = await get_questions_and_answers(chunk)
            if qa:
                text = ""
                for q in qa['qa']:
                    text += q.get('title', '') + '\n' + q.get('q_and_a', '') + '\n'
                if text:
                    chunks.append(text)

        # Move start position for next chunk
        start = max(start + 1, end)
    # print(f"Chunks with Q&A: {chunks}")
    return chunks

async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    """Extract title and summary from a documentation chunk."""
    system_prompt = (
        "You are an AI that extracts titles and summaries from documentation chunks.\n"
        "Return a JSON object with 'title' and 'summary' keys.\n"
        "For the title: If this seems like the start of a document, extract its title. "
        "If it's a middle chunk, derive a descriptive title.\n"
        "For the summary: Create a concise summary of the main points in this chunk.\n"
        "Keep both title and summary concise but informative."
    )
    try:
        response = await openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}
            ],
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error getting title and summary: {e}")
        return {"title": "Error processing title", "summary": "Error processing summary"}

async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model=embed_model,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * int(embed_dim)

async def process_chunk(chunk: str, chunk_number: int, url: str, source_name: str) -> ProcessedChunk:
    """Process a single chunk of text."""
    extracted = await get_title_and_summary(chunk, url)
    embedding = await get_embedding(chunk)
    metadata = {
        "source": source_name,
        "chunk_size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": urlparse(url).path
    }
    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title=extracted['title'],
        summary=extracted['summary'],
        content=chunk,
        metadata=metadata,
        embedding=embedding
    )

async def insert_chunk(chunk: ProcessedChunk, supabase_table: str):
    """Insert a processed chunk into Supabase."""
    try:
        data = {
            "url": chunk.url,
            "chunk_number": chunk.chunk_number,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding
        }
        result = supabase.table(supabase_table).insert(data).execute()
        print(f"Inserted chunk {chunk.chunk_number} for {chunk.url}")
        return result
    except Exception as e:
        print(f"Error inserting chunk: {e}")
        return None

def delete_chunk_by_url(url: str, supabase_table: str):
    """Delete chunks from Supabase by URL."""
    try:
        result = supabase.table(supabase_table).delete().eq("url", url).execute()
        print(f"Deleted chunk(s) with URL: {url}")
        return result
    except Exception as e:
        print(f"Error deleting chunk: {e}")
        return None

async def process_and_store_document(url: str, supabase_table: str, markdown: str, source_name: str, expand_details: str):
    """Process a document and store its chunks in parallel."""
    if expand_details == "Yes":
        print(f"Expanding details for URL: {url}")
        chunks = await chunk_text_with_qa(markdown)
    else:
        print(f"Non Expanding details for URL: {url}")
        chunks = await chunk_text(markdown)
    tasks = [
        process_chunk(chunk, i, url, source_name)
        for i, chunk in enumerate(chunks)
        if isinstance(chunk, str) and chunk.strip()
    ]
    processed_chunks = await asyncio.gather(*tasks)
    existing_chunks = supabase.table(supabase_table).select("id").eq("url", url).execute()
    if existing_chunks.data:
        delete_chunk_by_url(url, supabase_table)
    insert_tasks = [
        insert_chunk(chunk, supabase_table)
        for chunk in processed_chunks
    ]
    await asyncio.gather(*insert_tasks)

async def crawl_parallel(urls: List[str], supabase_table: str, source_name: str, expand_details: str, max_concurrent: int = 5):
    """Crawl multiple URLs in parallel with a concurrency limit."""
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()
    try:
        semaphore = asyncio.Semaphore(max_concurrent)
        async def process_url(url: str):
            async with semaphore:
                parsed_url = urlparse(url)
                if parsed_url.path.endswith('.pdf'):
                    print(f"Processing PDF: {url}")
                    try:
                        response = requests.get(url)
                        response.raise_for_status()
                        pdf_document = fitz.open(stream=response.content, filetype="pdf")
                        text = ""
                        for page_num in range(pdf_document.page_count):
                            page = pdf_document.load_page(page_num)
                            text += page.get_text()
                        pdf_document.close()
                        await process_and_store_document(url, supabase_table, text, source_name, expand_details)
                        print(f"Successfully processed PDF: {url}")
                    except Exception as e:
                        print(f"Failed to process PDF: {url} - Error: {e}")
                else:
                    result = await crawler.arun(
                        url=url,
                        config=crawl_config,
                        session_id="session1"
                    )
                    if result.success:
                        print(f"Successfully crawled: {url}")
                        await process_and_store_document(url, supabase_table, result.markdown_v2.raw_markdown, source_name, expand_details)
                    else:
                        print(f"Failed: {url} - Error: {result.error_message}")
        await asyncio.gather(*[process_url(url) for url in urls])
    finally:
        await crawler.close()

def get_urls_from_xml(url: str) -> List[str]:
    """Extract URLs from an XML sitemap."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        root = ElementTree.fromstring(response.content)
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]
        return urls
    except Exception as e:
        print(f"Error fetching sitemap: {e}")
        return []