import pandas as pd
import os
import json
import numpy as np
from sqlalchemy import create_engine
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from urllib.parse import quote_plus

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
load_dotenv()

# --- PostgreSQL Database Configuration ---
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
TABLE_NAME = os.getenv("TABLE_NAME")

# ==============================================================================
# 2. DATA AND MODEL LOADING (Done once at startup)
# ==============================================================================

def load_data_from_postgres(user, password, host, port, db, table) -> pd.DataFrame | None:
    """Connects to a PostgreSQL database and loads a table into a pandas DataFrame."""
    print(f"Connecting to PostgreSQL and loading table '{table}'...")
    try:
        # URL-encode the password to handle special characters safely
        encoded_password = quote_plus(password)
        
        # Construct the database URL. Azure PostgreSQL requires SSL by default.
        db_url = f'postgresql://{user}:{encoded_password}@{host}:{port}/{db}?sslmode=require'
        
        engine = create_engine(db_url)
        query = f'SELECT * FROM "{table}"'
        df = pd.read_sql_query(query, con=engine)
        
        print(f"\n✅ Loaded {len(df)} rows from table '{table}' into memory.")
        return df
        
    except Exception as e:
        print(f"❌ Could not connect to the database or load the table. Error: {e}")
        return None

def get_filters_from_llm(user_query: str, columns: list, llm: AzureChatOpenAI):
    """
    Uses an LLM to translate a natural language query into a complex JSON filter object.
    """
    prompt_template_string = """
    You are an expert at converting natural language questions into a structured JSON filter format.
    Analyze the user's query and map the identified entities to the most appropriate columns and operators.
    **1. Available Columns:**
    {columns}
    **2. Allowed Operators:**
    - "equals": For exact matches (case-insensitive for text).
    - "contains": For partial string matches (e.g., 'Los' in 'Los Angeles').
    - "greater_than": For numerical values greater than the specified value.
    - "less_than": For numerical values less than the specified value.
    - "in": For checking if a column's value is in a given list of items.
    **3. JSON Output Structure:**
    Your output MUST be a single JSON object with two keys: "relation" and "filters".
    - "relation": Can be either "AND" or "OR".
    - "filters": A list of filter objects, each with "column", "operator", and "value".
    **4. Rules:**
    - For numerical comparisons, the "value" must be a number.
    - Your response must be ONLY the JSON object.
    - If no filters are found, return {{"relation": "AND", "filters": []}}.
    **5. Examples:**
    Query: "Show me all the doctors"
    JSON:
    {{"relation": "AND", "filters": [{{"column": "Profession", "operator": "equals", "value": "doctor"}}]}}
    Query: "Show me nurses over 40 years old"
    JSON:
    {{"relation": "AND", "filters": [{{"column": "Profession", "operator": "equals", "value": "nurse"}}, {{"column": "Age", "operator": "greater_than", "value": 40}}]}}
    ---
    User Query: "{user_query}"
    JSON:
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template_string)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    
    response_str = chain.invoke({"columns": columns, "user_query": user_query})
    
    try:
        return json.loads(response_str)
    except (json.JSONDecodeError, TypeError):
        print("⚠️ Warning: LLM did not return valid JSON.")
        return None

# --- Define Request Body Structure ---
class QueryRequest(BaseModel):
    query: str

# --- Initialize FastAPI App and Load Global Objects ---
app = FastAPI(
    title="Natural Language Query API",
    description="An API that translates natural language questions into database queries.",
    version="1.0.0"
)

print("Initializing application...")
master_df = load_data_from_postgres(DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME, TABLE_NAME)
valid_columns = master_df.columns.tolist() if master_df is not None else []
# Langchain will automatically find the Azure environment variables loaded by dotenv
chat_llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0,
    max_tokens=400
)
print("✅ Application initialized and ready.")


# ==============================================================================
# 4. API ENDPOINT
# ==============================================================================

@app.post("/query")
async def handle_query(request: QueryRequest):
    """
    Accepts a natural language query and returns matching data from the database.
    """
    if master_df is None or not chat_llm:
        raise HTTPException(status_code=500, detail="Application is not initialized properly.")
    
    print(f"\nReceived query: '{request.query}'")
    
    filters_json = get_filters_from_llm(request.query, valid_columns, chat_llm)

    if not filters_json or not filters_json.get("filters"):
        raise HTTPException(status_code=400, detail={"message": "Could not determine filters.", "filters": filters_json})

    print(f"Identified filters: {filters_json}")

    try:
        relation = filters_json.get("relation", "AND").upper()
        filter_conditions = filters_json.get("filters", [])
        
        temp_df = master_df.copy()
        numeric_cols = {f['column'] for f in filter_conditions if f['operator'] in ['greater_than', 'less_than']}
        for col in numeric_cols:
            if col in temp_df.columns:
                temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')

        masks = []
        for f in filter_conditions:
            col, op, val = f.get("column"), f.get("operator"), f.get("value")
            if col not in temp_df.columns: continue
            str_col_accessor = temp_df[col].astype(str)

            if op == "equals": masks.append(str_col_accessor.str.lower() == str(val).lower())
            elif op == "contains": masks.append(str_col_accessor.str.contains(str(val), case=False, na=False))
            elif op == "in":
                val_list = [str(v).lower() for v in val] if isinstance(val, list) else [str(val).lower()]
                masks.append(str_col_accessor.str.lower().isin(val_list))
            elif op == "greater_than": masks.append(temp_df[col] > val)
            elif op == "less_than": masks.append(temp_df[col] < val)
        
        if not masks:
            return {"message": "No valid filters could be applied.", "results": []}

        if relation == "AND": combined_mask = np.logical_and.reduce(masks)
        else: combined_mask = np.logical_or.reduce(masks)
            
        results_df = master_df[combined_mask]
        
        results_json = json.loads(results_df.to_json(orient='records'))
        print(f"Found {len(results_json)} results.")
        return {"count": len(results_json), "results": results_json}
        
    except Exception as e:
        print(f"❌ An error occurred while applying filters: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")