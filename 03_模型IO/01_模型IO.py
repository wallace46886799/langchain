"""
欢迎来到LangChain实战课
https://time.geekbang.org/column/intro/100617601
作者 黄佳
"""
# 1.导入LangChain中的提示模板
from langchain import PromptTemplate
# 2.创建原始模板
template = """您是一位专业的鲜花店文案撰写员。\n
对于售价为 {price} 元的 {flower_name} ，您能提供一个吸引人的简短描述吗？
"""
# 3.根据原始模板创建LangChain提示模板
prompt = PromptTemplate.from_template(template) 
# 4.打印LangChain提示模板的内容
print(prompt)
""" 
    input_variables=['flower_name', 'price'] template='您是一位专业的鲜花店文案撰写员。\n\n对于售价为 {price} 元的 {flower_name} ，您能提供一个吸引人的简短描述吗？\n'
"""
# 设置OpenAI API Key
# import os
# os.environ["OPENAI_API_KEY"] = '你的OpenAI API Key'
from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量

# 5.导入LangChain中的OpenAI模型接口
from langchain_openai import ChatOpenAI
# 6.创建模型实例
# model = ChatOpenAI(model_name='gpt-3.5-turbo')
model = ChatOpenAI(model_name='gpt-3.5-turbo')
# 7.输入提示
input = prompt.format(flower_name="玫瑰", price='50')
print(input)

# 8.得到模型的输出
output = model.invoke(input)
# 9.打印输出内容
print(output.content)
