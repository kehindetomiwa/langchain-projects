from dotenv import load_dotenv
import os

load_dotenv()


def main():
    print("Hello from langchain-projects!")
    print(os.environ.get("OPENAI_API_KEY", "No OpenAI key found"))


if __name__ == "__main__":
    main()
