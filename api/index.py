from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Tuple, Optional
import json

from openai import OpenAI
import psycopg2
from datetime import datetime
import numpy as np
import os
import dotenv
dotenv.load_dotenv()
import re

# Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
DB_CONN = os.environ.get("DATABASE_URL")
TABLE_NAME = 'sitreps_2024'
EMBEDDING_MODEL = "text-embedding-ada-002"

DEFAULT_SYSTEM_INSTRUCTION = """You are an AI assistant specialized in cybersecurity incident analysis. Your task is to analyze the given query and related cybersecurity data, and provide a focused, relevant response. Follow these guidelines:
1. Analyze the user's query carefully to understand the specific cybersecurity concern or question.

2. Search through all provided relevant data columns to find information relevant to the query.

3. Conclude with a concise summary that highlights the key insights relevant to the user's query.

4. Structure your response to directly address the user's query, using only the most pertinent analysis points, and always include the id of the related sitrep in the response.

5. IMPORTANT: When discussing each record/incident, you MUST include the sitrep link that is provided in the data. The sitrep links are formatted as [Sitrep Link](URL) - make sure to include these exact links in your response for each record you analyze.

Your response should be informative and focused, ensuring clarity and relevance to the user's question while meeting the above guidelines."""

# Global variable to store system instruction
system_instruction = DEFAULT_SYSTEM_INSTRUCTION

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize FastAPI app
app = FastAPI(title="Cybersecurity Query API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str
    limit: Optional[int] = 5

class QueryResponse(BaseModel):
    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    message: str

class QueryAnalyzer:
    def analyze_query(self, query: str, available_columns: List[str]) -> Dict:
        """Analyze the user query to determine relevant columns and query intention"""
        try:
            current_instruction = system_instruction
            
            prompt = f"""
{current_instruction}

Please analyze this query: "{query}"

Available columns in the database: {', '.join(available_columns)}

Based on the above system instructions and considering cybersecurity context, extract and return a JSON object with the following information:
1. The most relevant columns for this query (only from the available columns list)
2. The main focus of the query from a cybersecurity perspective
3. Any specific data points or metrics mentioned that relate to security incidents
4. Any time frame mentioned
5. Any specific filtering criteria for security analysis

Format the response as a JSON object with these exact keys:
{{
    "relevant_columns": [], # list of column names from available_columns that are most relevant
    "query_focus": "", # main topic or purpose of the query from security perspective
    "specific_data_points": [], # list of specific security-related data points mentioned
    "time_frame": "", # time period mentioned, if any
    "filter_criteria": [] # any specific filtering criteria for security analysis
}}
"""
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": current_instruction},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=500,
                response_format={"type": "json_object"}
            )
            
            # Use json.loads instead of eval for safety
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error analyzing query: {str(e)}")
            return {
                "relevant_columns": [],
                "query_focus": "",
                "specific_data_points": [],
                "time_frame": "",
                "filter_criteria": []
            }

class DatabaseQuerier:
    def __init__(self):
        self.conn = None
        self.available_columns = []

    def connect_to_database(self):
        """Create connection to database"""
        try:
            if not DB_CONN:
                raise ValueError("Database connection string not found")
            self.conn = psycopg2.connect(DB_CONN)
            return True
        except Exception as e:
            print(f"Database connection error: {str(e)}")
            return False

    def close_connection(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

    def get_available_columns(self, table_name: str) -> List[str]:
        """Get list of available columns from the specified table"""
        if not self.conn:
            return []
        
        try:
            with self.conn.cursor() as cur:
                # First, ensure vector extension is available
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                self.conn.commit()
                
                cur.execute(f"""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = %s
                """, (table_name,))
                self.available_columns = [row[0] for row in cur.fetchall()]
                return self.available_columns
        except Exception as e:
            print(f"Error fetching columns: {str(e)}")
            return []

    def search_similar_records(self, query_embedding: List[float], relevant_columns: List[str], 
                             table_name: str, limit: int = 5) -> List[Dict]:
        """Search for similar records based on embedding"""
        if not self.conn:
            return []
        
        try:
            with self.conn.cursor() as cur:
                # Always include 'id' in the columns
                if "id" not in relevant_columns:
                    relevant_columns = ["id"] + relevant_columns
                columns_str = ", ".join(relevant_columns)
                
                cur.execute(f"""
                    SELECT {columns_str}
                    FROM {table_name}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (query_embedding, limit))
                
                columns = [desc[0] for desc in cur.description]
                results = cur.fetchall()
                
                return [dict(zip(columns, row)) for row in results]
        except Exception as e:
            print(f"Error searching records: {str(e)}")
            return []

def get_embedding(text: str) -> List[float]:
    """Get embedding for text using OpenAI's embedding model"""
    try:
        # Include system instruction context if not already present
        if not text.startswith("Context:"):
            text = f"Context: {system_instruction}\n{text}"
            
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {str(e)}")
        return []

def format_response(query: str, data: List[Dict], analysis: Dict) -> str:
    formatted_text = f"""
System Context: Using the following instruction for analysis:
{system_instruction}

Query: {query}

Analysis Focus: {analysis['query_focus']}
Time Frame: {analysis.get('time_frame', 'Not specified')}
Security Context: {analysis.get('specific_data_points', [])}

Retrieved Data:
"""
    base_url = "https://www.gradientcyber.net/cyber/cognitive/sitreps/"
    for idx, record in enumerate(data, 1):
        formatted_text += f"\nRecord {idx}:\n"
        for col, val in record.items():
            formatted_text += f"{col}: {val}\n"
        # Add the sitrep link for this record
        sitrep_id = record.get("id")
        if sitrep_id:
            formatted_text += f"[Sitrep Link]({base_url}{sitrep_id})\n"
    return formatted_text

def get_llm_response(query: str, formatted_data: str) -> str:
    """Get response from OpenAI based on the query and formatted data"""
    try:
        current_instruction = system_instruction
        
        prompt = f"""
As a data analyst, please analyze the following query and data to provide insights:

{formatted_data}

{current_instruction}

Format your response professionally and support your analysis with specific data points.

CRITICAL REQUIREMENT: For each record/incident you discuss, you MUST include the sitrep link that appears in the data as [Sitrep Link](URL). Do not just mention the Sitrep ID - include the actual clickable link provided in the data.
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": current_instruction},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error getting AI response: {str(e)}"

def check_environment():
    """Check if all required environment variables are set"""
    missing = []
    if not DB_CONN:
        missing.append("DATABASE_URL")
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    
    if missing:
        print(f"Missing required environment variables: {', '.join(missing)}")
        return False
    return True

def process_query(query: str, table_name: str, limit: int = 5) -> Tuple[List[Dict], Dict]:
    """Process a natural language query and return relevant data"""
    analyzer = QueryAnalyzer()
    querier = DatabaseQuerier()
    
    if not querier.connect_to_database():
        return [], {}
    
    try:
        # Get available columns
        available_columns = querier.get_available_columns(table_name)
        
        # Analyze the query with system instructions
        analysis = analyzer.analyze_query(query, available_columns)
        
        # Get embedding for the query with context
        query_with_context = f"""
Context: {system_instruction}
Query: {query}
Analysis Focus: {analysis['query_focus']}
"""
        query_embedding = get_embedding(query_with_context)
        
        # Search for similar records
        results = querier.search_similar_records(
            query_embedding,
            analysis['relevant_columns'],
            table_name,
            limit
        )
        
        return results, analysis
        
    finally:
        querier.close_connection()

def append_sitrep_links(results: List[Dict]) -> str:
    base_url = "https://www.gradientcyber.net/cyber/cognitive/sitreps/"
    links = []
    for record in results:
        sitrep_id = record.get("id")  # Use 'id' as the sitrep ID column
        if sitrep_id:
            links.append(f"- [Sitrep {sitrep_id}]({base_url}{sitrep_id})")
    if links:
        return "\n\n**Related Sitrep Links:**\n" + "\n".join(links)
    return ""

@app.get("/")
def root():
    return {"message": "Cybersecurity Query API is running"}

@app.get("/health", response_model=HealthResponse)
def health_check():
    return {"status": "healthy", "message": "API is up and running"}

@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    """Main API endpoint for processing cybersecurity queries"""
    query = request.query.strip()
    limit = request.limit

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        # Check environment variables first
        if not check_environment():
            raise HTTPException(status_code=500, detail="Server configuration error")

        # Process the query
        results, analysis = process_query(query, TABLE_NAME, limit)
        
        if results:
            formatted_data = format_response(query, results, analysis)
            ai_response = get_llm_response(query, formatted_data)
            
            return QueryResponse(
                success=True,
                data={
                    "response": ai_response
                }
            )
        else:
            return QueryResponse(
                success=True,
                data={
                    "response": "No data found matching your query. Try rephrasing your question or use different keywords."
                }
            )
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/available-columns")
def get_available_columns():
    """Get available database columns"""
    try:
        querier = DatabaseQuerier()
        if not querier.connect_to_database():
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        try:
            columns = querier.get_available_columns(TABLE_NAME)
            return {"success": True, "columns": columns}
        finally:
            querier.close_connection()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching columns: {str(e)}")

def main():
    """Main entry point - for local development"""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# For Vercel deployment
if __name__ == "__main__":
    main()
