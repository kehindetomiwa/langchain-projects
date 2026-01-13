from typing import List, Union

from dotenv import load_dotenv
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
from langchain_core.tools import Tool, tool
from langchain_openai import ChatOpenAI

from callbacks import AgentCallbackHandler

load_dotenv()


@tool
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters."""
    text = text.strip("'\n").strip('"')
    return len(text)


def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"{tool_name} is not found in tools")


def main():
    print("Hello bind_tools LangChain!")
    tools = [get_text_length]

    llm = ChatOpenAI(
        temperature=0,
        callbacks=[AgentCallbackHandler()],
    )
    
    # Bind tools to the LLM
    agent_model = llm.bind_tools(tools)
    
    intermediate_step = []
    
    # Initial message for the agent
    system_message = "You are a helpful assistant. Use the tools available to answer questions accurately."
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": "What is the text length of 'Dog' in characters?"},
    ]
    
    agent_step = None

    while True:
        # Get response from model with tools
        response = agent_model.invoke(messages)
        
        # Check if there are tool calls
        if response.tool_calls:
            # Process each tool call
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_input = tool_call["args"]
                tool_id = tool_call["id"]
                
                tool_to_use = find_tool_by_name(tools, tool_name)
                observation = tool_to_use.func(**tool_input) if isinstance(tool_input, dict) else tool_to_use.func(tool_input)
                
                print(f"Tool: {tool_name}")
                print(f"Input: {tool_input}")
                print(f"Observation: {observation}")
                
                # Add assistant response and tool result to messages
                messages.append({"role": "assistant", "content": response.content, "tool_calls": response.tool_calls})
                messages.append({
                    "role": "tool",
                    "content": str(observation),
                    "tool_call_id": tool_id,
                })
        else:
            # No tool calls, this is the final response
            print("Final Answer:", response.content)
            break


if __name__ == "__main__":
    main()
