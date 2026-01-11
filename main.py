from pyexpat.errors import messages

from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

# from tavily import TavilyClient

# tavily = TavilyClient()


# @tool
# def search_tool(query: str) -> str:
#     """
#     Tool Searches the internet for the given query and returns the results.
#     Args:
#         query: The search query.
#     Returns:
#         The search results.
#     """
#
#     print("Searching for:", query)
#     return tavily.search(query)


llm = ChatOpenAI()
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello from langchain-projects!")
    # result = agent.invoke({"messages": HumanMessage(content="What is weather in Tokyo?")})
    result = agent.invoke(
        {
            "messages": HumanMessage(
                content="search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details?"
            )
        }
    )
    print("Agent result:", result)


if __name__ == "__main__":
    main()
