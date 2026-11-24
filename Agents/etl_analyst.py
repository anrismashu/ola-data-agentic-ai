import os
import re
import sys
import base64
import zlib
import requests
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Utils.llm_pick import pick_llm
from Utils.etl_tools import ETLTools
from Models.schema import ETLAgentSchema
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage , ToolMessage
from langgraph.graph import StateGraph ,START , END
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

#............................. Helpers .................................................

def content_to_text(content) -> str:
    """Turn an LLM message's .content into a plain string.

    Newer Gemini models (via langchain-google-genai) can return .content as a list of
    blocks like [{"type": "text", "text": "..."}] instead of a str, so string methods
    such as .strip() fail on it. This handles both shapes.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type", "text") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


def extract_code(text: str) -> str:
    """Return only the code from an LLM reply, with or without markdown fences."""
    match = re.search(r"```[\w+-]*[ \t]*\n(.*?)(?:```|\Z)", text, re.DOTALL)
    return (match.group(1) if match else text).strip()

#............................. ETL Agent Class .................................................

@tool
def extract_load_tool(url:str,output_folder:str,format:str)-> str: 
    """this tool extract data from api url and load it into the desired location destination output_folder.
    Args:
        url (str): The API URL to extract data from.
        output_folder (str): The destination folder to save the extracted data.
        format (str): The format to save the extracted data (csv, json, parquet).
    Returns:
        str: a message indicating the success or failure of the extraction process.
    """
    etl_tools = ETLTools()
    result = etl_tools.extract_data(url, output_folder, format)
    return result

@tool
def transform_load_tool(input_file_path:str,output_folder:str,output_format:str,user_question:str)-> str:
    """this tool transforms and loads data from the specified file path to the desired location destination output_folder.
    Args:
        input_file_path (str): The path to the input file to be transformed and loaded.
        output_folder (str): The destination folder to save the transformed data.
        output_format (str): The format to save the transformed data (csv, json, parquet).
        user_question (str): What the user wants done to the data (the transformation to apply).
    Returns:
        str: a message indicating the success or failure of the transformation and loading process.
    """
    etl_tools = ETLTools()  
    # THE FIX: Only pass the input_file_path to get the context
    top_3_rows = etl_tools.transform_load_context(input_file_path)  
    llm = pick_llm("high")   
    prompt = f"""
You are a python data analyst. Reply with ONLY the pandas code that performs the ETL operation the user asked for.
Do not provide any explanation or any other text. The code must be directly executable in python.

Load the data from this file into a pandas dataframe called df: {input_file_path}
The top 3 rows of the file are as follows:
{top_3_rows}

Write the code to transform the data and save it in the folder {output_folder} in {output_format} format.
Create the folder first if it does not exist.

Here is the user question: {user_question}
"""
    # Gemini may return a list of content blocks, so normalise to text before cleaning it up
    response = content_to_text(llm.invoke(prompt).content)
    pandas_code = extract_code(response)
    results = etl_tools.execute_code(pandas_code)
    return f"The data is transformed and saved successfully at {output_folder} in {output_format} format. Here is the pandas code used:\n{pandas_code}\n\nExecution Result:\n{results}"
#toolkit
tools = [extract_load_tool, transform_load_tool]
llm = pick_llm("high")
llm_bind = llm.bind_tools(tools)
    
#....agent graph

SYSTEM_PROMPT = """You are a python data analyst. You have access to tools that can extract, transform, and load data:
- extract_load_tool: pulls data from an API url and saves it to a folder.
- transform_load_tool: transforms a file that is already saved and writes out the result.

Analyze the user's question and use only the tool(s) it actually needs. Do not transform the data
unless the user asked for a transformation. Once the requested operation is done, tell the user
what happened (including where the data was saved) and stop. Do not call any more tools."""

def llm_node(state: ETLAgentSchema):
    # Send the real message objects (not a stringified list) so the model sees
    # its own tool calls and the tool results in the proper structure.
    response = llm_bind.invoke([SystemMessage(content=SYSTEM_PROMPT)] + list(state.messages))
    
    # LangGraph will automatically append this to the list
    return {"messages": [response]}

def tool_node(state:ETLAgentSchema):
    """_summary_
    this node is responsible for executing the tool based on the user question and the context provided. It will analyze the user question and decide 
    which tool to use for the ETL operation. After executing the tool, it will update the state with the results and inform the user about the success
    or failure of the operation.
    Args:
        state (ETLAgentSchema): _description_
    """
    tool_calls = state.messages[-1].tool_calls
    tools_result = []
    tools_by_name = {tool.name: tool for tool in tools}
    
    for tool_call in tool_calls:
        tool = tools_by_name[tool_call['name']]
        observation = tool.invoke(tool_call['args'])
        tools_result.append(ToolMessage(content=observation, tool_call_id=tool_call['id'], name=tool_call['name']))
        
    return {"messages": tools_result}

# nodes and edges for the graph

etl_analayst_graph = StateGraph(ETLAgentSchema)
etl_analayst_graph.add_node("llm_node", llm_node)
etl_analayst_graph.add_node("tool_node", tool_node)

etl_analayst_graph.add_edge(START, "llm_node")

def is_tool_call(state:ETLAgentSchema):
    tool_calls = state.messages[-1].tool_calls
    if tool_calls:
        return "tool_node"
    else:
        return "end"
    
etl_analayst_graph.add_conditional_edges(
    "llm_node",is_tool_call,
    {
        "tool_node":"tool_node",
        "end":END
    }
)
etl_analayst_graph.add_edge("tool_node","llm_node")
etl_analyst = etl_analayst_graph.compile()
if __name__ == "__main__":

    # Render Graph Image via Kroki
    try:
        mermaid_code = etl_analyst.get_graph().draw_mermaid()
        compressed = zlib.compress(mermaid_code.encode("utf-8"), 9)
        b64_string = base64.urlsafe_b64encode(compressed).decode("ascii")
        url = f"https://kroki.io/mermaid/png/{b64_string}"

        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            with open("etl_analyst_graph.png", "wb") as f:
                f.write(res.content)
            print("Graph saved as 'etl_analyst_graph.png'")
    except Exception as e:
        print(f"Skipping graph render: {e}")
    
    # response = etl_analyst.invoke(
    #     {"messages":[HumanMessage(content="I want to extract the data form the api endpoint 'https://pokeapi.co/api/v2/pokemon'. I want to save it data/extract folder in the csv format ")]}
    # )
    
    # final_messages = response.get("messages", [])
    # if final_messages:
    #     print("\n--- FINAL ANSWER ---")
    #     print(content_to_text(final_messages[-1].content))
    
    # print("\n--- FULL STATE ---")
    # print(response)
    
    response = etl_analyst.invoke(
        {"messages":[HumanMessage(content=f"""
           i want to transform the data stored in the 'C:\EveryPython\Ola_Data_\Ola_Data_Agentic_Ai\data\extract\extracted_data.csv' file
            and save the transfrom data in the  'C:\EveryPython\Ola_Data_\Ola_Data_Agentic_Ai\data\tranform' folder in the csv format.
            The transformation should filter the data to show bulbasaur pokemon only                                            
        """)]})
    final_messages = response.get("messages", [])
    print("\n--- FINAL ANSWER ---")
    print(content_to_text(final_messages[-1].content))
    print(response)