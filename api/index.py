from flask import Flask, request, jsonify, abort, Response
from flask_cors import CORS
from typing import Dict, List, Tuple, Optional
import json
import os
import dotenv
import re
from datetime import datetime

from openai import OpenAI
import psycopg2

# Load environment variables
dotenv.load_dotenv()
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
"""

# Global variable to store system instruction
system_instruction = DEFAULT_SYSTEM_INSTRUCTION

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Flask app
app = Flask(__name__)

# Add CORS support
CORS(app)

class QueryAnalyzer:
    def analyze_query(self, query: str, available_columns: List[str]) -> Dict:
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
                max_tokens=500
            )
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
        try:
            if not DB_CONN:
                raise ValueError("Database connection string not found")
            self.conn = psycopg2.connect(DB_CONN)
            return True
        except Exception as e:
            print(f"Database connection error: {str(e)}")
            return False

    def close_connection(self):
        if self.conn:
            self.conn.close()

    def get_available_columns(self, table_name: str) -> List[str]:
        if not self.conn:
            return []
        try:
            with self.conn.cursor() as cur:
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
        if not self.conn:
            return []
        try:
            with self.conn.cursor() as cur:
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
    try:
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
    formatted = f"""
System Context:
{system_instruction}

Query: {query}

Analysis Focus: {analysis['query_focus']}
Time Frame: {analysis.get('time_frame', 'Not specified')}
Security Context: {analysis.get('specific_data_points', [])}

Retrieved Data:
"""
    base_url = "https://www.gradientcyber.net/cyber/cognitive/sitreps/"
    for idx, record in enumerate(data, 1):
        formatted += f"\nRecord {idx}:\n"
        for col, val in record.items():
            formatted += f"{col}: {val}\n"
        if record.get("id"):
            formatted += f"[Sitrep Link]({base_url}{record['id']})\n"
    return formatted

def get_llm_response(query: str, formatted_data: str) -> str:
    try:
        current_instruction = system_instruction
        prompt = f"""
As a data analyst, please analyze the following query and data:

{formatted_data}

{current_instruction}

Ensure for each incident discussed, the sitrep link is included as [Sitrep Link](URL).
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

def get_llm_response_stream(query: str, formatted_data: str):
    """Generator function for streaming LLM response"""
    try:
        current_instruction = system_instruction
        prompt = f"""
As a data analyst, please analyze the following query and data:

{formatted_data}

{current_instruction}

Ensure for each incident discussed, the sitrep link is included as [Sitrep Link](URL).
"""
        
        # Send initial metadata
        yield f"data: {json.dumps({'type': 'start', 'message': 'Starting analysis...'})}\n\n"
        
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": current_instruction},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1000,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                yield f"data: {json.dumps({'type': 'content', 'content': content})}\n\n"
        
        # Send completion signal
        yield f"data: {json.dumps({'type': 'complete'})}\n\n"
        
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

def check_environment():
    missing = []
    if not DB_CONN:
        missing.append("DATABASE_URL")
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if missing:
        print(f"Missing env vars: {', '.join(missing)}")
        return False
    return True

def process_query(query: str, table_name: str, limit: int = 5) -> Tuple[List[Dict], Dict]:
    analyzer = QueryAnalyzer()
    querier = DatabaseQuerier()

    if not querier.connect_to_database():
        return [], {}

    try:
        available_columns = querier.get_available_columns(table_name)
        analysis = analyzer.analyze_query(query, available_columns)
        query_with_context = f"""
Context: {system_instruction}
Query: {query}
Analysis Focus: {analysis['query_focus']}
"""
        query_embedding = get_embedding(query_with_context)
        results = querier.search_similar_records(query_embedding, analysis['relevant_columns'], table_name, limit)
        return results, analysis
    finally:
        querier.close_connection()

@app.route("/")
def root():
    return jsonify({"status": "healthy", "message": "Cybersecurity Query API is running"})

@app.route("/health")
def health_check():
    return jsonify({"status": "healthy", "message": "API is up and running"})

@app.route("/available-columns")
def get_available_columns():
    try:
        querier = DatabaseQuerier()
        if not querier.connect_to_database():
            abort(500, description="Database connection failed")
        try:
            columns = querier.get_available_columns(TABLE_NAME)
            return jsonify({"success": True, "columns": columns})
        finally:
            querier.close_connection()
    except Exception as e:
        abort(500, description=f"Error fetching columns: {str(e)}")

@app.route("/query", methods=["POST"])
def query_endpoint():
    if not request.json:
        abort(400, description="Request must be JSON")
    
    query = request.json.get("query", "").strip()
    limit = request.json.get("limit", 5)

    if not query:
        abort(400, description="Query cannot be empty")

    try:
        if not check_environment():
            abort(500, description="Missing required environment variables")

        results, analysis = process_query(query, TABLE_NAME, limit)

        if results:
            formatted_data = format_response(query, results, analysis)
            ai_response = get_llm_response(query, formatted_data)
            return jsonify({"success": True, "data": {"response": ai_response}})
        else:
            return jsonify({"success": True, "data": {"response": "No data found matching your query."}})
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        abort(500, description=f"Error: {str(e)}")

@app.route("/query-stream", methods=["POST"])
def query_stream_endpoint():
    """Streaming version of the query endpoint"""
    if not request.json:
        abort(400, description="Request must be JSON")
    
    query = request.json.get("query", "").strip()
    limit = request.json.get("limit", 5)

    if not query:
        abort(400, description="Query cannot be empty")

    try:
        if not check_environment():
            abort(500, description="Missing required environment variables")

        results, analysis = process_query(query, TABLE_NAME, limit)

        if results:
            formatted_data = format_response(query, results, analysis)
            
            def generate():
                yield f"data: {json.dumps({'type': 'query_processed', 'query': query})}\n\n"
                yield from get_llm_response_stream(query, formatted_data)
            
            return Response(
                generate(),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'Cache-Control'
                }
            )
        else:
            def generate_no_data():
                yield f"data: {json.dumps({'type': 'complete', 'content': 'No data found matching your query.'})}\n\n"
            
            return Response(
                generate_no_data(),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'Cache-Control'
                }
            )
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        def generate_error():
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        
        return Response(
            generate_error(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Cache-Control'
            }
        )

if __name__ == "__main__":
    app.run(debug=True)
