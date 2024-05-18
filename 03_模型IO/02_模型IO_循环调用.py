'''
欢迎来到LangChain实战课
https://time.geekbang.org/column/intro/100617601
作者 黄佳
'''
# 导入LangChain中的提示模板
from langchain import PromptTemplate
# 创建原始模板
template = """您是一位专业的鲜花店文案撰写员。\n
对于售价为 {price} 元的 {flower_name} ，您能提供一个吸引人的简短描述吗？
"""
# 根据原始模板创建LangChain提示模板
prompt = PromptTemplate.from_template(template) 
# 打印LangChain提示模板的内容
print(prompt)

# 设置OpenAI API Key
from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量

# 导入LangChain中的OpenAI模型接口
from langchain_openai import ChatOpenAI
# 创建模型实例
# model = ChatOpenAI(model_name='gpt-3.5-turbo')
model = ChatOpenAI(model_name='gpt-3.5-turbo')

# 导入LangChain中的OpenAI模型接口
# from langchain_openai import ChatOpenAI
# # 创建模型实例
# model = ChatOpenAI(model_name='gpt-3.5-turbo')

# 多种花的列表
flowers = ["玫瑰", "百合", "康乃馨"]
prices = ["50", "30", "20"]

# 生成多种花的文案
for flower, price in zip(flowers, prices):
    # 使用提示模板生成输入
    input_prompt = prompt.format(flower_name=flower, price=price)
    print(input_prompt)
    # 得到模型的输出
    output = model.invoke(input_prompt)

    # 打印输出内容
    print(output.content)
