# pip install langchain streamlit langchain_openai python-dotenv    
## Langchain for the purpose to building chatbot logic
## Streamlit for Designing web based UI

# Libraries to import
import streamlit as st 
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    from langchain.chat_models import ChatOpenAI
try:
    from langchain_classic.chains import ConversationChain
    from langchain_classic.memory import ConversationBufferMemory
except ImportError:
    from langchain.chains import ConversationChain
    from langchain.memory import ConversationBufferMemory
import os 
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set your openAI API key securely (checks Streamlit secrets and environment variables)
raw_key = (
    st.secrets.get("OPENAI_API_KEY") or st.secrets.get("API") if hasattr(st, "secrets") and st.secrets else None
) or os.getenv("OPENAI_API_KEY") or os.getenv("API_Key") or os.getenv("API")

# Ignore placeholder values
openai_api_key = raw_key if raw_key and "your_openai_api_key_here" not in raw_key else None

# Streamlit UI
st.set_page_config(page_title="LLM Chatbot", page_icon="🤖")
st.title("LangChain Chatbot")

if not openai_api_key:
    st.error("⚠️ OpenAI API key not found. Please add your key in `.env` or `.streamlit/secrets.toml`.")
    st.stop()

# Initialize LLM
llm = ChatOpenAI(
    openai_api_key = openai_api_key,
    temperature = 0.7,
    model_name = "gpt-3.5-turbo"
)

# Set up memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory()

# Conversation Chain
conversation = ConversationChain(
    llm = llm,
    memory = st.session_state.memory,
    verbose = False
)

user_input = st.text_input("You:", key="input")

if user_input:
    try:
        response = conversation.predict(input = user_input)
        st.session_state.memory.chat_memory.add_user_message(user_input)
        st.session_state.memory.chat_memory.add_ai_message(response)
        st.write(f"**Bot:** {response}")
    except Exception as e:
        st.error(f"⚠️ OpenAI API Error: {e}")

# Show history (Optional)
if st.checkbox("Show Chat History"):
    for message in st.session_state.memory.chat_memory.messages:
        role = "You" if message.type == "human" else "Bot"
        st.markdown(f"**{role}:**{message.content}")


# To run this app
# 1. Open your terminal or command prompt
# 2. Navigate to the directory where you saved the file
# 3. Run the command: streamlit run chatbot.py 