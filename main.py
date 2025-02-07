"""
メインプログラム
"""
import os
import csv

from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma

load_dotenv()

def read_problem_csv_file():
    """ 質問文CSVファイルの読み込み """
    problem_list = []

    # CSVファイル読み込み
    with open("./query.csv", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)

        problem_list = [row[1] for row in reader]

    return problem_list


def make_chat_model():
    """ Azure OpenAIのチャットモデルを生成 """

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

    return model


def make_embeddings():
    """ Azure OpenAIのエンベディングを生成 """

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS"),
    )

    return embeddings


def make_retriever():
    """ リトリーバーを生成 """

    # Embedding
    embeddings = make_embeddings()

    # ベクターデータベースの読み込み
    vectorstore = Chroma(persist_directory="./chroma", embedding_function=embeddings)

    # 検索結果上位３位までを取得
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    return retriever


def make_prompt():
    """ プロンプトを生成 """

    human_message_template = HumanMessagePromptTemplate.from_template('''\
    以下の文脈を参考にして、質問に簡潔に1文で回答してください。

    ### 文脈
    {context}

    ### 質問
    {question}
    ''')

    chat_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage("質問に対して適切な情報が文脈に含まれる場合のみ回答してください。"),
            human_message_template,
        ]
    )

    return chat_prompt


def make_output_parser():
    """ パーサーを生成 """

    return StrOutputParser()


def main():
    """ メイン処理 """

    # queryファイル読み込み
    problem_list = read_problem_csv_file()

    # Retriever
    retriever = make_retriever()

    # インプット（引数）
    input_dict = {
        "context": retriever,
        "question": RunnablePassthrough(),
    }

    # Prompt
    chat_prompt = make_prompt()

    # Model
    model = make_chat_model()

    # Output Parser
    output_parser = make_output_parser()

    # Chain
    chain = (
        input_dict
        | chat_prompt
        | model
        | output_parser
    )

    response = chain.invoke(problem_list[1])
    print(response)

if __name__ == "__main__":
    main()
