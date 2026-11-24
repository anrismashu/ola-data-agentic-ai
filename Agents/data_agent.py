import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Utils.llm_pick import pick_llm
from Models.schema import DataAgentSchema, RouterSchema
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from etl_analyst import content_to_text, etl_analyst
from sql_Analyst import sql_analyst

# ......................... Initialization .........................

llm = pick_llm("low")
llm_router = llm.with_structured_output(RouterSchema)

CHAT_SYSTEM_PROMPT = """You are Ola Data Agentic AI, a friendly and concise AI data assistant.
You have two specialist agents:
1. SQL Analyst — answers read-only questions over 61,000+ Ola ride-sharing records (users, vehicles, rides, payments, ratings) in PostgreSQL.
2. ETL Analyst — pulls data from public API URLs and transforms/saves CSV or JSON files with Pandas.

Use the conversation history to remember details the user shares (like their name) and reply naturally and briefly."""


def format_recent_history(messages: list, max_prev: int = 4) -> str:
    """Format up to `max_prev` earlier messages (excluding the newest one) as compact context."""
    earlier = messages[:-1][-max_prev:]
    lines = []
    for msg in earlier:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        text = content_to_text(msg.content).strip().replace("\n", " ")
        if len(text) > 250:
            text = text[:250] + "..."
        lines.append(f"{role}: {text}")
    return "\n".join(lines)


# ......................... Node Functions .........................

def router_node(state: DataAgentSchema) -> dict:
    latest = content_to_text(state.messages[-1].content)
    history = format_recent_history(state.messages, max_prev=3)
    history_block = f"Recent Conversation:\n{history}\n\n" if history else ""

    prompt = f"""
    You are a routing supervisor. Decide who should handle the user's latest message:
    - Output 'sql' if the user is asking about Ola database records (rides, users, payments, ratings, vehicles, fares, cities, drivers) or following up on a previous database result.
    - Output 'etl' if the user wants to extract data from an API URL, or transform/filter/save a file (CSV, JSON, Parquet).
    - Output 'chat' for greetings, small talk, personal questions (like recalling their name), or questions about what this assistant can do.

    {history_block}Latest User Message: {latest}
    """
    route_response = llm_router.invoke(prompt)
    return {"route_response": route_response.answer.strip().lower()}


def chat_node(state: DataAgentSchema) -> dict:
    recent_msgs = list(state.messages[-8:])
    reply = content_to_text(
        llm.invoke([SystemMessage(content=CHAT_SYSTEM_PROMPT)] + recent_msgs).content
    )
    return {"messages": [AIMessage(content=reply)]}


def etl_node(state: DataAgentSchema) -> dict:
    response = etl_analyst.invoke({"messages": list(state.messages[-6:])})
    final_message = response.get("messages", [])[-1]
    return {"messages": [final_message]}


def sql_node(state: DataAgentSchema) -> dict:
    latest = content_to_text(state.messages[-1].content)
    history = format_recent_history(state.messages, max_prev=4)

    if history:
        question_with_context = (
            f"Recent conversation (for follow-up context only):\n{history}\n\n"
            f"Current question to answer with SQL: {latest}"
        )
    else:
        question_with_context = latest

    response = sql_analyst.invoke({"user_question": question_with_context})
    final_message = response.get("messages", [])[-1]
    sql_query = response.get("generated_sql_query", "")
    if sql_query:
        final_message = AIMessage(
            content=final_message.content,
            additional_kwargs={"sql_query": sql_query},
        )
    return {"messages": [final_message]}


# ......................... Routing Logic .........................

def route_condition(state: DataAgentSchema) -> str:
    route = state.route_response
    if "sql" in route:
        return "sql_node"
    if "etl" in route:
        return "etl_node"
    return "chat_node"


# ......................... Graph Definition .........................

supervisor_graph = StateGraph(DataAgentSchema)
supervisor_graph.add_node("router_node", router_node)
supervisor_graph.add_node("chat_node", chat_node)
supervisor_graph.add_node("etl_node", etl_node)
supervisor_graph.add_node("sql_node", sql_node)

supervisor_graph.add_edge(START, "router_node")
supervisor_graph.add_conditional_edges(
    "router_node",
    route_condition,
    {
        "chat_node": "chat_node",
        "etl_node": "etl_node",
        "sql_node": "sql_node",
    },
)
supervisor_graph.add_edge("chat_node", END)
supervisor_graph.add_edge("etl_node", END)
supervisor_graph.add_edge("sql_node", END)

main_agent = supervisor_graph.compile()