# AI Chatbot Project - Scraper Tool

This project is an AI-powered web scraping tool that can crawl and process documents from URLs, including PDF files. It uses FastAPI for the backend, Streamlit for the frontend, and integrates with OpenAI and Supabase for processing and storing data.

## Features

- Crawl multiple URLs in parallel with a concurrency limit.
- Handle both regular web pages and PDF documents.
- Extract text from PDF documents using PyMuPDF.
- Process and store document chunks in Supabase.
- Extract titles and summaries using OpenAI's GPT-4.
- Generate embedding vectors for text chunks using OpenAI.

## Requirements

- Python 3.8+
- FastAPI
- Streamlit
- PyMuPDF
- OpenAI API
- Supabase

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/sahaki/Scraper_website_tool
    cd Scraper_website_tool
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

3. Create a [.env](example.env) file in the project root and add your OpenAI and Supabase credentials:
    ```env
    OPENAI_API_KEY=your_openai_api_key
    SUPABASE_URL=your_supabase_url
    SUPABASE_SERVICE_KEY=your_supabase_service_key
    BACKEND_PORT=8000
    OPENAI_MODEL=gpt-4o-mini
    ```

4. Set up the database table in Supabase:
    - Open the Supabase Dashboard.
    - Navigate to the SQL Editor.
    - Create a new query.
    - Copy and paste the content of the [site_pages.sql] file into the SQL Editor.
    - Click on the "Run" button to execute the SQL code.

## Usage

### Backend

1. Run the FastAPI backend:
    ```sh
    python chatbot_backend.py
    ```

2. The backend will be available at `http://localhost:8000`.

### Frontend

1. Run the Streamlit frontend:
    ```sh
    streamlit run app.py
    ```

2. The frontend will be available at `http://localhost:8501`.

## API Endpoints

### POST /crawl

Crawl and process documents from URLs.

#### Request Body

```json
{
    "scrape_type": "XML" | "URL",
    "url": "string",
    "supabase_table": "string",
    "source_name": "string"
}