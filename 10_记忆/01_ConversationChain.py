'''欢迎来到LangChain实战课
https://time.geekbang.org/column/intro/100617601
作者 黄佳'''
# 设置OpenAI API密钥
# import os
# os.environ["OPENAI_API_KEY"] = 'Your Key'

from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量

# 导入所需的库
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain

# 初始化大语言模型
llm = ChatOpenAI(
    temperature=0.5,
    model_name="gpt-3.5-turbo"
)

# 初始化对话链
conv_chain = ConversationChain(llm=llm)

# 打印对话的模板
print(conv_chain.prompt.template)
