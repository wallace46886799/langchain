'''欢迎来到LangChain实战课
https://time.geekbang.org/column/intro/100617601
作者 黄佳'''

import os
# os.environ["OPENAI_API_KEY"] = '你的OpenAI API Key'

from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量

from langchain.chat_models import ChatOpenAI
chat = ChatOpenAI(model="gpt-4",
                    temperature=0.8,
                    max_tokens=60)
from langchain.schema import (
    HumanMessage,
    SystemMessage
)
messages = [
    SystemMessage(content="你是一个很棒的智能助手"),
    HumanMessage(content="请给我的花店起个名")
]
response = chat(messages)
print(response)
