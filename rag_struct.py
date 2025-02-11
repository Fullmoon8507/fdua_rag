"""
ChromaDBを用いてベクターデータベースを作成する。
"""
import os
import time
import pdfplumber

from dotenv import load_dotenv

# from langchain_community.document_loaders import PyMuPDFLoader
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

load_dotenv()

# load_pdf_list = [f"./documents/{i}.pdf" for i in range(1, 20)]
load_pdf_list = [
    "./documents/1.pdf",
    "./documents/2.pdf",
    "./documents/3.pdf",
    "./documents/4.pdf",
    "./documents/5.pdf",
    "./documents/6.pdf",
    "./documents/7.pdf",
    "./documents/8.pdf",
    "./documents/9.pdf",
    "./documents/10.pdf",
    "./documents/11.pdf",
    "./documents/12.pdf",
    "./documents/13.pdf",
    "./documents/14.pdf",
    "./documents/15.pdf",
    "./documents/16.pdf",
    "./documents/17.pdf",
    "./documents/18.pdf",
    "./documents/19.pdf",
]

# splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Embedding
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS"),
)


def extract_text_from_pdf(pdf_path):
    """pdfplumber を用いて PDF からテキストを抽出する"""
    texts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            texts.append(page.extract_text() or "")

    return "\n".join(texts)


if __name__ == "__main__":

    # ベクターデータベースの保存先
    PERSIST_DIRECTORY = "./chroma"
    vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)

    for pdf_file in load_pdf_list:

        print(f"{pdf_file}のEmbeddingを実行")

        # PDF読み込み
        # loader = PyMuPDFLoader(pdf_file)
        # data = loader.load()
        pdf_text = extract_text_from_pdf(pdf_file)
        document = Document(page_content=pdf_text, metadata={"source": pdf_file})

        # テキスト分割
        chunks = text_splitter.split_documents([document])

        # ChromaDBに保存。また、永続化する。
        vectorstore.add_documents(chunks)

        # HTTPステータス500回避のため、Sleepを入れる。
        time.sleep(30)

    print("すべてのPDFをChromaDBに保存しました。")
    for pdf in load_pdf_list:
        print(pdf)
