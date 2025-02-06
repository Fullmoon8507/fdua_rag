"""
メインプログラム
"""
import os
import csv

from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains.retrieval_qa.base import RetrievalQA

load_dotenv()

# Model
model = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    temperature=os.getenv("AZURE_OPENAI_EMPERATURE"),
    max_tokens=os.getenv("AZURE_OPENAI_AX_TOKENS"),
    top_p=os.getenv("AZURE_OPENAI_OP_P"),
    frequency_penalty=os.getenv("AZURE_OPENAI_REQUENCY_PENALTY"),
    presence_penalty=os.getenv("AZURE_OPENAI_RESENCE_PENALTY"),
)

# Embedding
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS"),
)

def read_problem_csv_file():
    """ メイン処理 """
    problem_list = []

    # CSVファイル読み込み
    with open("./query.csv", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)

        problem_list = [row[1] for row in reader]

    return problem_list

def main():
    """ メイン処理 """

    # queryファイル読み込み
    problem_list = read_problem_csv_file()

    vectorstore = Chroma(persist_directory="./chroma", embedding_function=embeddings)
    retriever = vectorstore.as_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm=model,
        retriever = retriever,
        return_source_documents=True,
    )

    user_input = problem_list[1]
    response = qa_chain.invoke(user_input)
    print(response["result"])

if __name__ == "__main__":
    main()
