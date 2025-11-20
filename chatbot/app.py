from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os 
from dotenv import load_dotenv

load_dotenv()
# OpenAI API Key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Langsmith tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ.setdefault("LANGCHAIN_PROJECT", "chatbot-dev")

# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user's question in a friendly and helpful manner."),
        ("user", "Question: {input}"),
    ]
)

# Stremlit framework
st.title("Llm-test-app")
input_text = st.text_input("Enter your question here:")

# OpenAi LLM
llm = ChatOpenAI(model="gpt-5-nano", temperature=0)
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    response = chain.invoke({"input": input_text})
    st.write(response)