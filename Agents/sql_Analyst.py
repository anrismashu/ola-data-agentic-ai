import base64
import os
import sys
import zlib
import requests
import re 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from Models.schema import AgentSchema, JudgeSchema
from Utils.database import DataBaseUtils
from Utils.llm_pick import pick_llm

# ......................... Node Functions .........................

# Cached DB connection & schema (loaded once, reused on every question)
_db_instance = None
_cached_schema = None

def get_db() -> DataBaseUtils:
    global _db_instance
    if _db_instance is None:
        conn_details = {
            "host": os.environ.get("host", "localhost"),
            "port": os.environ.get("port", 5432),
            "database": os.environ.get("database", "postgres"),
            "user": os.environ.get("user", "postgres"),
            "password": os.environ.get("password", "4141"),
        }
        _db_instance = DataBaseUtils(conn_details)
    return _db_instance

def get_cached_schema() -> str:
    global _cached_schema
    if not _cached_schema:
        _cached_schema = get_db().schema_detils("public")
    return _cached_schema

# ......................... Node Functions .........................

def prompt_querry_context(state: AgentSchema) -> dict:
    schema_info = get_cached_schema()
    prompt = f"""
You are an SQL analyst agent. Convert the user's natural language query into a valid Postgres SQL query.
Schema Details:
{schema_info}

User's Original Query: {state.user_question}

Instructions:
- Return ONLY the executable SQL query.
- Do NOT include markdown blocks, explanations, or comments.
- Unless explicitly asked, always LIMIT the output to 10 rows.
- For revenue or payment totals, always filter `payment_status = 'completed'` unless the user specifically asks about failed, pending, or all payments.
- For ride metrics, filter `status = 'completed'` unless the user asks about cancelled or all rides.
- This is a historical dataset: for relative time phrases like "last month" or "last 2 months", anchor to `(SELECT MAX(payment_time) FROM public.payments)` or `(SELECT MAX(requested_at) FROM public.rides)` instead of `CURRENT_DATE` or `NOW()`.
- When listing top drivers or riders, JOIN `public.users` to include `first_name` and `last_name` alongside the metrics instead of returning only raw IDs.
"""
    return {
        "curated_ques": state.user_question.strip(),
        "prompt_querry_context": prompt,
    }


def generate_sql(state: AgentSchema) -> dict:
    llm = pick_llm("medium")
    sql_content = llm.invoke(state.prompt_querry_context).content
    if isinstance(sql_content, list):
        sql_content = sql_content[0]["text"]

    sql_content = sql_content.strip()
    if sql_content.startswith("```"):
        sql_content = sql_content.split("\n", 1)[-1]
    if sql_content.endswith("```"):
        sql_content = sql_content.rsplit("```", 1)[0]

    return {"generated_sql_query": sql_content.strip()}


def is_safe_sql(state: AgentSchema) -> dict:
    """Instant deterministic guardrail: allows only single read-only SELECT/WITH queries."""
    query = state.generated_sql_query.strip().rstrip(";")
    upper = query.upper()

    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        return {"is_safe": "No", "comments": "Only SELECT queries are permitted."}
    if ";" in query:
        return {"is_safe": "No", "comments": "Multiple SQL statements are not allowed."}

    forbidden = r"\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|GRANT|REVOKE|EXECUTE|COPY)\b"
    if re.search(forbidden, upper):
        return {"is_safe": "No", "comments": "Query contains a forbidden write/DDL keyword."}

    return {"is_safe": "Yes", "comments": "Verified read-only SELECT query."}


def cancled_sql(state: AgentSchema) -> dict:
    explanation = f"The generated SQL query is not safe to execute. Reason: {state.comments}"
    return {
        "final_answer": explanation,
        "messages": [AIMessage(content=explanation)],
    }


def execute_sql(state: AgentSchema) -> dict:
    result = get_db().execute_query(state.generated_sql_query)
    return {"sql_querry_execution_result": result}


def represent_final_answer(state: AgentSchema) -> dict:
    llm = pick_llm("low")
    prompt = f"""
Provide a concise, friendly final answer to the user based on the database results.
Question: {state.user_question}
SQL Executed: {state.generated_sql_query}
Query Results: {state.sql_querry_execution_result}
"""
    final_content = llm.invoke(prompt).content
    if isinstance(final_content, list):
        final_content = final_content[0]["text"]

    return {
        "final_answer": final_content,
        "messages": [AIMessage(content=final_content)],
    }

# ......................... Graph Definition .........................

sql_agent_graph = StateGraph(AgentSchema)

sql_agent_graph.add_node("Prompt Query Context", prompt_querry_context)
sql_agent_graph.add_node("Generate SQL Query", generate_sql)
sql_agent_graph.add_node("Is Safe SQL", is_safe_sql)
sql_agent_graph.add_node("Cancled SQL", cancled_sql)
sql_agent_graph.add_node("Execute SQL Query", execute_sql)
sql_agent_graph.add_node("Represent Final Answer", represent_final_answer)

sql_agent_graph.add_edge(START, "Prompt Query Context")
sql_agent_graph.add_edge("Prompt Query Context", "Generate SQL Query")
sql_agent_graph.add_edge("Generate SQL Query", "Is Safe SQL")

def is_safe_sql_condition(state: AgentSchema) -> str:
    if state.is_safe.lower() == "yes":
        return "Execute SQL Query"
    return "Cancled SQL"

sql_agent_graph.add_conditional_edges("Is Safe SQL", is_safe_sql_condition)
sql_agent_graph.add_edge("Cancled SQL", END)
sql_agent_graph.add_edge("Execute SQL Query", "Represent Final Answer")
sql_agent_graph.add_edge("Represent Final Answer", END)

sql_analyst = sql_agent_graph.compile()

# ......................... Execution .........................

if __name__ == "__main__":
    # Render Graph Image via Kroki
    try:
        mermaid_code = sql_analyst.get_graph().draw_mermaid()
        compressed = zlib.compress(mermaid_code.encode("utf-8"), 9)
        b64_string = base64.urlsafe_b64encode(compressed).decode("ascii")
        url = f"[https://kroki.io/mermaid/png/](https://kroki.io/mermaid/png/){b64_string}"

        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            with open("sql_analyst_graph.png", "wb") as f:
                f.write(res.content)
            print("Graph saved as 'sql_analyst_graph.png'")
    except Exception as e:
        print(f"Skipping graph render: {e}")

    # Invoke Agent
    initial_input = {
        "user_question": "What are the different types of payment methods we have in our database?"
    }

    result = sql_analyst.invoke(initial_input)
    
    # FUTURE ISSUE FIX: Safely retrieve dictionary keys using .get() to prevent KeyErrors on aborted/unsafe logic branches
    print("\n--- FINAL ANSWER ---")
    print(result.get("final_answer", "No final answer generated."))
    
    print("\n****************** MESSAGES ******************")
    for msg in result.get('messages', []):
        print(f"{type(msg).__name__}: {msg.content}\n")
        
    print("****************** EXECUTION RESULT ******************")
    print(result.get('sql_querry_execution_result', 'Execution Skipped (Query potentially unsafe)'))
    
    print("\n****************** GENERATED SQL ******************")
    print(result.get('generated_sql_query', 'None generated'))