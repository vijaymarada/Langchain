from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langserve import add_routes
from langchain_community.llms import Ollama
import uvicorn 
import os
from dotenv import load_dotenv 

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
#os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")


app = FastAPI(
    title="Langchain API with FastAPI",
    description="This is a Langchain API built with FastAPI",
    version="1.0"
)

add_routes(app,
           ChatOpenAI(),
           path="/openai"
)
model=ChatOpenAI()
llm=Ollama(model="llama3")


prompt1= ChatPromptTemplate.from_template("What is complete description about {topic}")
prompt2= ChatPromptTemplate.from_template("What are the key points of {topic}")


add_routes(app,
           prompt1|model,           
           path="/description",
)

add_routes(app,
           prompt2|llm,
           path="/keypoints",
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)


    # run this with  python app.py