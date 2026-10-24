from docx2txt import process
from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv # It is used to store secret stuff like API keys or configuration values

load_dotenv()
class AgentState(TypedDict):
    message : List[HumanMessage]

llm = GoogleGenerativeAI("gemini-2.5-flash")

def pocess(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    print(f"\nAI: {response.content}")
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

# This is for single input and answer process code
# user_input = input("Enter:")
# agent.invoke({"message": [HumanMessage(content = user_input)]}) 

# This is the multiple questions and answers process code
user_input = input("Enter: ")
while user_input != "exit":
    agent.invoke({"messages" : [HumanMessage(content=user_input)]})
    user_input = input("Enter: ")
