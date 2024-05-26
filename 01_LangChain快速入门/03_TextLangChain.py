'''欢迎来到LangChain实战课
https://time.geekbang.org/column/intro/100617601
作者 黄佳'''

import os
# os.environ["OPENAI_API_KEY"] = '你的OpenAI API Key'

from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量

from langchain_openai import ChatOpenAI
model_name="gpt-3.5-turbo"
llm = ChatOpenAI(base_url=os.environ.get("OPENAI_API_BASE"),
    model=model_name,
    temperature=0.8,
    max_tokens=60)
from loguru import logger
response = llm.invoke("请给我的花店起个名")
logger.debug(response)
