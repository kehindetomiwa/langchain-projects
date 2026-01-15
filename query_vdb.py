# create a simple script that queries the vector database

import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


load_dotenv()


def query_vector_db(query: str):
    db_name = "vector_db"
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    vectorstore = Chroma(persist_directory=db_name, embedding_function=embeddings)
    count = vectorstore._collection.count()
    print(f"Vector store contains {count} vectors.")
    # retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


if __name__ == "__main__":

    documents = query_vector_db("user_query")
