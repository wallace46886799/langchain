'''欢迎来到LangChain实战课
https://time.geekbang.org/column/intro/100617601
作者 黄佳'''
from openai import OpenAI # 导入OpenAI
import os
# 设置OpenAI API Key
from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量
client = ChatOpenAI(api_key=os.environ.get("OPENAI_API_KEY"),
  base_url=os.environ.get("OPENAI_API_BASE"))

prompt_text = "您是一位专业的鲜花店文案撰写员。对于售价为{}元的{}，您能提供一个吸引人的简短描述吗？" # 设置提示

flowers = ["玫瑰", "百合", "康乃馨"]
prices = ["50", "30", "20"]

# 循环调用Text模型的Completion方法，生成文案
for flower, price in zip(flowers, prices):
   prompt = prompt_text.format(price, flower)
   print(prompt)
   response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
           {
               "role": "user",
               "content": prompt,
           }
        ],
        max_tokens=100,
    )
   print(response.choices[0].message.content) # 输出文案
