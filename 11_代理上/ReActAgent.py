'''欢迎来到LangChain实战课
https://time.geekbang.org/column/intro/100617601
作者 黄佳'''
# 设置OpenAI和SERPAPI的API密钥
import os
# os.environ["OPENAI_API_KEY"] = 'Your OpenAI API Key'
from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量

# 加载所需的库
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain_openai import ChatOpenAI

# 初始化大模型
llm = ChatOpenAI(temperature=0)

# 设置工具
tools = load_tools(["serpapi", "llm-math"], llm=llm)

# 初始化Agent
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# 跑起来
agent.invoke("目前市场上玫瑰花的平均价格是多少？如果我在此基础上加价15%卖出，应该如何定价？")
