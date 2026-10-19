# pip install langchain streamlit langchain_openai langchain-classic python-dotenv
import os
import streamlit as st
from dotenv import load_dotenv

# Try importing ChatOpenAI
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    from langchain.chat_models import ChatOpenAI

# Try importing ConversationChain and Memory (supports newer LangChain and classic)
try:
    from langchain_classic.chains import ConversationChain
    from langchain_classic.memory import ConversationBufferMemory
except ImportError:
    from langchain.chains import ConversationChain
    from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

# Streamlit UI Setup
st.set_page_config(page_title="LLM Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 LangChain Chatbot")

# Sidebar for API Key & Model Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    env_key = (
        (st.secrets.get("OPENAI_API_KEY") or st.secrets.get("API") if hasattr(st, "secrets") and st.secrets else None)
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("API_Key")
        or os.getenv("API")
        or ""
    )
    if "your_openai_api_key_here" in env_key:
        env_key = ""

    openai_api_key = st.text_input(
        "OpenAI API Key",
        value=env_key,
        type="password",
        help="Enter your OpenAI API key (starts with sk-). You can also set it in .env"
    )

    model_name = st.selectbox(
        "Select Model",
        ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o"],
        index=0
    )

    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)

    if st.button("🗑️ Clear Chat History"):
        st.session_state.memory = ConversationBufferMemory(return_messages=True)
        st.rerun()

# Ensure API Key is available
if not openai_api_key:
    st.warning("⚠️ OpenAI API key not found. Please enter your API key in the sidebar or add it to your `.env` file.")
    st.stop()

# Initialize LLM
try:
    llm = ChatOpenAI(
        api_key=openai_api_key,
        temperature=temperature,
        model=model_name
    )
except Exception as e:
    st.error(f"Failed to initialize LLM: {e}")
    st.stop()

# Set up ConversationBufferMemory in session state
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

# Conversation Chain
conversation = ConversationChain(
    llm=llm,
    memory=st.session_state.memory,
    verbose=False
)

# Display existing messages from memory
for msg in st.session_state.memory.chat_memory.messages:
    role = "user" if msg.type == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# Chat input prompt
user_input = st.chat_input("Type your message here...")

if user_input:
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # conversation.predict automatically records input & output in memory
                response = conversation.predict(input=user_input)
                st.markdown(response)
            except Exception as e:
                st.error(f"⚠️ OpenAI API Error: {e}")

# To run this app:
# streamlit run chatbot.py