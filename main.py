"""
メインプログラム
"""
import os
import csv
import time
import tiktoken

from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma

load_dotenv()


class TokenLimitOutputParser(BaseOutputParser):
    """ トークン数を超える文字列を切り詰めるパーサー """

    def parse(self, text: str) -> str:

        tokenizer = tiktoken.encoding_for_model("gpt-4-turbo")
        max_token = 54

        # トークン数を取得
        tokens = tokenizer.encode(text)

        if len(tokens) > max_token:

            # 54トークン以内に切り詰める
            trimmed_text = tokenizer.decode(tokens[:max_token])
            return trimmed_text

        return text


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
        temperature=os.getenv("AZURE_OPENAI_TEMPERATURE"),
        max_tokens=os.getenv("AZURE_OPENAI_MAX_TOKENS"),
        top_p=os.getenv("AZURE_OPENAI_TOP_P"),
        frequency_penalty=os.getenv("AZURE_OPENAI_FREQUENCY_PENALTY"),
        presence_penalty=os.getenv("AZURE_OPENAI_PRESENCE_PENALTY"),
    )

    return model


def make_virtual_answer_model():
    """ Azure OpenAIのチャットモデルを生成(仮想回答用) """

    model = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        temperature=os.getenv("AZURE_OPENAI_TEMPERATURE"),
        top_p=os.getenv("AZURE_OPENAI_TOP_P"),
        frequency_penalty=os.getenv("AZURE_OPENAI_FREQUENCY_PENALTY"),
        presence_penalty=os.getenv("AZURE_OPENAI_PRESENCE_PENALTY"),
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


def search_from_vectorstore(search_key):
    """ VectorStoreから文章検索 """

    # Embedding
    embeddings = make_embeddings()

    # ベクターデータベースの読み込み
    vectorstore = Chroma(persist_directory="./chroma", embedding_function=embeddings)

    # 検索
    document_list = vectorstore.similarity_search(query=search_key, k=3)

    # 必要な情報のみを抽出
    result_list = []
    for document in document_list:
        result_list.append((document.page_content, document.metadata['source']))

    # promptに埋め込むようのフォーマットに変換
    document = ""
    for doc in result_list:
        document += f"docName={doc[1]},\n"
        document += f"docContent={doc[0]}\n"

    # return result_list
    return document


def make_prompt():
    """ プロンプトを生成 """

    few_shot_examples = [
        {
            "input": "A株式会社が売り上げ上位３つのセグメントは？",
            "output": "改修セグメント、医療用・産業用セグメント、官公庁セグメント",
        },
        {
            "input": "B株式会社が海外に持っている拠点数は？",
            "output": "36拠点",
        },
        {
            "input": "2023下期の売り上げは？",
            "output": "わかりません",
        },
    ]

    few_shot_template = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}")
        ]
    )

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=few_shot_template,
        examples=few_shot_examples,
    )
    human_message_template = HumanMessagePromptTemplate.from_template('''\
    以下の文脈を参考にして、質問に簡潔に数単語で回答してください。

    ### 文脈
    {context}

    ### 質問
    {question}
    ''')

    chat_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage("質問に対して適切な情報が文脈に含まれる場合のみ回答し、含まれない場合は「分かりません」と回答して。回答は必ず54トークン以内に短縮して簡潔に述べること。"),
            few_shot_prompt,
            human_message_template,
        ]
    )

    return chat_prompt


def make_hypothetical_prompt():
    """ 仮想回答用のプロンプト """

    hypothetical_prompt = ChatPromptTemplate.from_template('''\
    次の質問に回答する一文を書いてください。

    質問：{question}
    ''')

    return hypothetical_prompt


def make_output_parser():
    """ パーサーを生成 """

    # return StrOutputParser()
    return TokenLimitOutputParser()


def main():
    """ メイン処理 """

    # queryファイル読み込み
    problem_list = read_problem_csv_file()

    # インプット（引数）
    input_dict = {
        "context": RunnablePassthrough(),
        "question": RunnablePassthrough(),
    }

    # Prompt
    chat_prompt = make_prompt()
    hypothetical_prompt = make_hypothetical_prompt()

    # Model
    model = make_chat_model()
    virtual_answer_model = make_virtual_answer_model()

    # Output Parser
    output_parser = make_output_parser()

    # Chain
    hypothetical_chain = (
        hypothetical_prompt
        | virtual_answer_model
        | StrOutputParser()
    )

    hyde_rag_chain = (
        input_dict
        | chat_prompt
        | model
        | output_parser
    )

    responses = []
    for i, problem in enumerate(problem_list):

        # 仮想の回答を生成
        virtual_answer = hypothetical_chain.invoke(problem)

        # 仮想の回答を検索キーに、ベクターデータベースから文章検索
        document = search_from_vectorstore(virtual_answer)

        print(f"{i+1}番目 問題　　： {problem}")
        print(f"{i+1}番目 仮想回答： {virtual_answer}")

        response = hyde_rag_chain.invoke({"context": document, "question": problem})
        responses.append(response)

        print(f"{i+1}番目 回答　　： {response}")

        time.sleep(1)

    # チャット結果を応募形式に沿ったcsvファイルに出力
    with open("submit/predictions.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for i, response in enumerate(responses):
            writer.writerow([i, response])


if __name__ == "__main__":
    main()