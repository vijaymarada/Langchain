import requests
import streamlit as st

def get_openai_response(prompt):    
    response = requests.post("http://localhost:8000/description/invoke", 
                             json={"input": {"topic": prompt}})
    return response.json()['output']['content']


def get_llama_response(prompt):    
    response = requests.post("http://localhost:8000/keypoints/invoke", 
                             json={"input": {"topic": prompt}})
    print(response.json())
    return response.json()['output']

st.title("Langchain API with FastAPI")
input_text=st.text_input("[OPENAPI] Write a description about")
input_text2=st.text_input("[OLLAMA] Write a keypoints about")

if input_text:
    st.write(get_openai_response(input_text))
if input_text2:
    st.write(get_llama_response(input_text2))

# Run this with  - streamlit run .\client.py