'''欢迎来到LangChain实战课
https://time.geekbang.org/column/intro/100617601
作者 黄佳'''

from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量

# 以下代码采用最新版本的langchain和openai  pip install langchain, pip install openai, pip install langchain-openai
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5, max_tokens=200)
text = llm.invoke("请给我写一句情人节红玫瑰的中文宣传语")
from loguru import logger
logger.debug("提示语：{}","请给我写一句情人节红玫瑰的中文宣传语")
logger.debug("生成的宣传语：{}",text.content)
