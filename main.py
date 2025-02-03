"""
メインプログラム
"""
import os
import csv

from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def read_problem_csv_file():
    """ メイン処理 """
    problem_list = []

    # CSVファイル読み込み
    with open("./query.csv", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)

        problem_list = [row[1] for row in reader]

def main():
    """ メイン処理 """
    model = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    )

    human_message_template = HumanMessagePromptTemplate.from_template("{user_input}")
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage("質問に対して簡潔に回答してください。"),
            human_message_template,
        ]
    )

    chain = chat_prompt | model | StrOutputParser()
    response =  chain.invoke({"user_input": "こんにちは"})
    print(response)

if __name__ == "__main__":
    # read_problem_csv_file()
    main()
