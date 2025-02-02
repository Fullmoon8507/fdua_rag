"""
メインプログラム
"""
import csv

from langchain_openai import AzureChatOpenAI

def read_problem_csv_file():
    """ メイン処理 """
    problem_list = []

    # CSVファイル読み込み
    with open("./query.csv", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)

        problem_list = [row[1] for row in reader]

def main2():
    """ メイン処理 """
    llm = AzureChatOpenAI(
        azure_endpoint="AAA",
        api_key="BBB",
        api_version="CCC",
        deployment_name="DDD",
    )

    response = llm.invoke("こんにちは")
    print(response.content)

if __name__ == "__main__":
    # read_problem_csv_file()
    main2()
