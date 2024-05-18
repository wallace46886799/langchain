'''欢迎来到LangChain实战课
https://time.geekbang.org/column/intro/100617601
作者 黄佳'''
# 设置OpenAI和SERPAPI的API密钥
import os
from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量
os.environ["SERPAPI_API_KEY"] = '068163c315bc7a57a8d7c5321f79367aaa594dd3580661c7abeda29f507e52ea'  # 068163c315bc7a57a8d7c5321f79367aaa594dd3580661c7abeda29f507e52ea

# 试一试LangChain的Debug和Verbose，看看有何区别
import langchain
langchain.debug = True
langchain.verbose = True

# 配置日志输出
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# 跟踪与openai的通信
import openai
# openai.api_key = os.environ["OPENAI_API_KEY"]
openai.log = "debug"

# 加载所需的库
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI

# 初始化大模型
llm = ChatOpenAI(temperature=0)

# 设置工具
tools = load_tools(["serpapi", "llm-math"], llm=llm)

# 初始化Agent
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# 跑起来
agent.run("目前市场上玫瑰花的平均价格是多少？如果我在此基础上加价15%卖出，应该如何定价？")
