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
os.environ["SERPAPI_API_KEY"] = '068163c315bc7a57a8d7c5321f79367aaa594dd3580661c7abeda29f507e52ea'  # 068163c315bc7a57a8d7c5321f79367aaa594dd3580661c7abeda29f507e52ea



from langchain_openai import ChatOpenAI, SerpAPIWrapper
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

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
self_ask_with_search.run(
    "使用玫瑰作为国花的国家的首都是哪里?"  
)
