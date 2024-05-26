'''欢迎来到LangChain实战课
https://time.geekbang.org/column/intro/100617601
作者 黄佳'''
# 设置OpenAI和SERPAPI的API密钥
import os
# os.environ["OPENAI_API_KEY"] = 'Your Key'
# os.environ["SERPAPI_API_KEY"] = 'Your Key'
from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量


from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents.tools import Tool
from langchain.chains import LLMMathChain

search = SerpAPIWrapper()
llm = ChatOpenAI(temperature=0)
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math"
    ),
]
model = ChatOpenAI(temperature=0)
planner = load_chat_planner(model)
executor = load_agent_executor(model, tools, verbose=True)
agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)

agent.invoke("在纽约，100美元能买几束玫瑰?")
