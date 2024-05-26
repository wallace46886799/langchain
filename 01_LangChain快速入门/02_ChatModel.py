'''欢迎来到LangChain实战课
https://time.geekbang.org/column/intro/100617601
作者 黄佳'''

import os
from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量


# os.environ["OPENAI_API_KEY"] = '你的OpenAI API Key'

# import openai

# response = openai.ChatCompletion.create(
#   model="gpt-4",
#   messages=[
#         {"role": "system", "content": "You are a creative AI."},
#         {"role": "user", "content": "请给我的花店起个名"},
#     ],
#   temperature=0.8,
#   max_tokens=60
# )

# print(response['choices'][0]['message']['content'])

# print(response.choices)


from openai import OpenAI
client = OpenAI(
  api_key=os.environ.get("OPENAI_API_KEY"),
  base_url=os.environ.get("OPENAI_API_BASE"))
model_name = "gpt-3.5-turbo"
response = client.chat.completions.create(
  messages=[
        {"role": "system", "content": "You are a creative AI."},
        {"role": "user", "content": "请给我的花店起个名"},
    ],
    temperature=0.2,
    max_tokens=100,
    model=model_name,
)

from loguru import logger
logger.debug("OpenAI的Chat模型：{}；返回的花店名称为：{}".format(model_name,response.choices[0].message.content))
