import requests
import streamlit as st

def get_short_story(subject):
    response = requests.post(
        "http://localhost:8000/short-story/invoke",
        json={"input": {"subject": subject}},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    return data["output"].get("content") or data["output"]

st.title("Langchain Server")
input_text = st.text_input("Enter a subject for a short story: ") 

if input_text:
    st.write(get_short_story(input_text))