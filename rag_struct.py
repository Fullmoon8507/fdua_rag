"""
ChromaDBを用いてベクターデータベースを作成する。
"""
import os

from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()

if __name__ == "__main__":

    # PDF読み込み
    loader = PyMuPDFLoader("./documents/1.pdf")
    data = loader.load()

    # テキスト分割
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(data)

    # Embedding
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS"),
    )

    # ChromaDBに保存。また、永続化する。
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma")
