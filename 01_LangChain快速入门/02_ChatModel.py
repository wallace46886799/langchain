'''欢迎来到LangChain实战课
https://time.geekbang.org/column/intro/100617601
作者 黄佳'''

from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量

# import os
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
client = OpenAI(base_url="https://api.chatanywhere.tech/v1")

response = client.chat.completions.create(model="gpt-4",
  messages=[
        {"role": "system", "content": "You are a creative AI."},
        {"role": "user", "content": "请给我的花店起个名"},
    ],
  temperature=0.8,
  max_tokens=60
)

print(response.choices[0].message.content)
