# pip install langchain streamlit langchain_openai python-dotenv    
## Langchain for the purpose to building chatbot logic
## Streamlit for Designing web based UI

# Libraries to import
import streamlit as st 
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import os 

# Set your openAI API key securely 
openai_api_key = st.secrets["API"] if "API" in st.secrets else os.getenv("API_Key")

# Initialize LLM
llm = ChatOpenAI(
    
)
