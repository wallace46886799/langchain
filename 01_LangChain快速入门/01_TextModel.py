"""
欢迎来到LangChain实战课
https://time.geekbang.org/column/intro/100617601
作者 黄佳
"""

from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量

import os
# os.environ["OPENAI_API_KEY"] = '你的OpenAI API Key'
# os.environ["OPENAI_API_BASE"] = "https://api.chatanywhere.cn/v1"
# import openai
# # openai.api_key = '你的OpenAI API Key'

# response = openai.Completion.create(
#   model="gpt-3.5-turbo",
#   temperature=0.5,
#   max_tokens=100,
#   prompt="请给我的花店起个名")

# print(response.choices[0].text.strip())
# TODO.
from openai import OpenAI
client = OpenAI(
  api_key=os.environ.get("OPENAI_API_KEY"),
  base_url=os.environ.get("OPENAI_API_BASE"))

# response = client.completions.create(
#   model="gpt-3.5-turbo",
#   temperature=0.5,
#   max_tokens=100,
#   prompt="请给我的花店起个名")
model_name = "gpt-3.5-turbo"
# response = client.completions.create(
#   model="gpt-3.5-turbo",
#   temperature=0.5,
#   max_tokens=100,
#   prompt="请给我的花店起个名")
# 由于Chatanywhere的限制，我们只能使用Chat模型
response = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "请给我的花店起个名",
        }
    ],
    temperature=0.2,
    max_tokens=100,
    model=model_name,
)
from loguru import logger
logger.debug("OpenAI的Text模型：{}；返回的花店名称为：{}".format(model_name,response.choices[0].message.content))
