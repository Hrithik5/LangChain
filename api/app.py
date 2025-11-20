from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langserve import add_routes

import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

app = FastAPI(
    title="Langchain Server",
    version="1.0.0",
    description="A simple Langchain server",
)

add_routes(
    app,
    ChatOpenAI(),
    path="/openai",
)

model = ChatOpenAI(model="gpt-5-nano", temperature=0)
prompt1 = ChatPromptTemplate.from_template("Write a short story about a {subject} with 100 words.")
chain1 = prompt1 | model

add_routes(
    app,
    chain1,
    path="/short-story",
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)

