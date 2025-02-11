"""
ChromaDBを用いてベクターデータベースを作成する。
"""
import os
import time
import fitz
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
    doc = fitz.open(pdf_path)

    # ページ単位でループ
    for page_num in range(len(doc)):

        # PyMuPDFでテキスト抽出
        page = doc[page_num]
        text = page.get_text("text")

        # テキストが取得できた場合(空白だけのデータ無視)
        if text.strip():
            texts.append(text)
        
        # PyMuPDFで抽出できなかった場合、pdfplumberで補完
        else:
            print("pdfplumberでPDF読み込み")

            # pdfplumberでPDFを開く
            with pdfplumber.open(pdf_path) as pdf:
                table_text = []

                # 現在のページを取得
                pdf_page = pdf.pages[page_num]

                # 表があれば取得
                tables = pdf_page.extract_table()

                # テーブルの各行に対して、セルの値を | で区切って 1 行のテキストに変換
                if tables:
                    print("テーブルのテキスト化あり")

                    for row in tables:
                        # None を 空文字("") に置き換える
                        cleaned_row = [cell if cell is not None else "" for cell in row]
                        table_text.append(" | ".join(cleaned_row))
                
                # すべての行を改行で結合
                texts.append("\n".join(table_text))

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
        time.sleep(10)

    print("すべてのPDFをChromaDBに保存しました。")
    for pdf in load_pdf_list:
        print(pdf)
