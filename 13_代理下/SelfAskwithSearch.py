'''欢迎来到LangChain实战课
https://time.geekbang.org/column/intro/100617601
作者 黄佳'''
# 设置OpenAI和SERPAPI的API密钥
# import os
# os.environ["OPENAI_API_KEY"] = 'Your Key'
# os.environ["SERPAPI_API_KEY"] = 'Your Key'

import os
from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量

from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents.tools import Tool

# llm = ChatOpenAI(temperature=0)

llm = ChatOpenAI(temperature=0.5)

search = SerpAPIWrapper()
tools = [
    Tool(
        name="Intermediate Answer", 
        func=search.run,
        description="useful for when you need to ask with search",
    )
]

self_ask_with_search = initialize_agent(
    tools, llm, agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=True, handle_parsing_errors=True
)
self_ask_with_search.invoke("使用玫瑰作为国花的国家的首都是哪里?")
