from dotenv import load_dotenv
from importlib.metadata import version

load_dotenv()

from langchain_core import __version__ as core_version
lg_version = version("langgraph")
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic 
import os

print(f"langchain-core version: {core_version}")
print(f"langgraph version: {lg_version}")



def main():

    # Testing Google gen-ai
    google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if google_key:
        print("\n--- Testing Google Gemini ---")
        llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", max_tokens=500, temperature=0.5, api_key=google_key)
        response = llm.invoke([("human", "Hello, how are you?")])
        print(response.content)
    else:
        print("\n[Skipping Google Gemini]: GOOGLE_API_KEY is not set in .env")

    # Testing OpenAI
    # openai_key = os.getenv("OPENAI_API_KEY")
    # if openai_key:
    #     print("\n--- Testing OpenAI ---")
    #     llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=500, temperature=0.5, api_key=openai_key)
    #     response = llm.invoke([("human", "Hello, how are you?")])
    #     print(response.content)
    # else:
    #     print("\n[Skipping OpenAI]: OPENAI_API_KEY is not set in .env")

    # Testing Anthropic
    # anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    # if anthropic_key:
    #     print("\n--- Testing Anthropic ---")
    #     llm = ChatAnthropic(model="claude-3-haiku-20240307", max_tokens=500, temperature=0.5, api_key=anthropic_key)
    #     response = llm.invoke([("human", "Hello, how are you?")])
    #     print(response.content)
    # else:
    #     print("\n[Skipping Anthropic]: ANTHROPIC_API_KEY is not set in .env")

    # print("\nSetup complete!")

if __name__ == "__main__":
    main()