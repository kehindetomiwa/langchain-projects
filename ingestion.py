import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()


if __name__ == "__main__":
    print("Ingesting...")

    db_name = "vector_db"

    loader = TextLoader("mediumlog1.txt")
    document = loader.load()

    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    print("ingesting...")
    if os.path.exists(db_name):
        Chroma(
            persist_directory=db_name, embedding_function=embeddings
        ).delete_collection()

    vectorstore = Chroma.from_documents(
        documents=texts, embedding=embeddings, persist_directory=db_name
    )

    print("finish")
