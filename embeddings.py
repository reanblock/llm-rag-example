from dotenv import load_dotenv
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

load_dotenv(override=True)

# Test different embedding models
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")