"""
Here we define all the reasoning agents that decide which tools to call and when to call them.
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_tavily import TavilySearch


load_dotenv()

@tool
def triple(num: float) -> float:
    """
    returns the triple of a number
    param num: a number to be tripled
    return: triple of the number
    """
    return float(num) * 3

tools = [triple, TavilySearch(max_results=1)]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)


