'''欢迎来到LangChain实战课
https://time.geekbang.org/column/intro/100617601
作者 黄佳'''

# import os
# os.environ["OPENAI_API_KEY"] = '你的OpenAI API Key'

from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量

from langchain.llms import OpenAI
llm = OpenAI(
    base_url="https://api.chatanywhere.tech/v1",
    model="gpt-3.5-turbo-instruct",
    temperature=0.8,
    max_tokens=60)
response = llm.predict("请给我的花店起个名")
print(response)
