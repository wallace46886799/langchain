# 设置OpenAI的API密钥
# import os
# os.environ["OPENAI_API_KEY"] = 'Your OpenAI Key'
from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量

# 导入文档加载器模块，并使用TextLoader来加载文本文件
from langchain_community.document_loaders import TextLoader
loader = TextLoader('../02_文档QA系统/OneFlower/花语大全.txt', encoding='utf8')

from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

# 使用VectorstoreIndexCreator来从加载器创建索引
from langchain.indexes import VectorstoreIndexCreator
index = VectorstoreIndexCreator(embedding=embeddings).from_loaders([loader])
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model_name='gpt-3.5-turbo')
# 定义查询字符串, 使用创建的索引执行查询
query = "紫罗兰的花语是什么？"
result = index.query(query,llm=llm)
print(result)  # 打印查询结果


# 替换成你所需要的工具
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
from langchain.vectorstores import Chroma

index_creator = VectorstoreIndexCreator(
    vectorstore_cls=Chroma,
    embedding=OpenAIEmbeddings(),
    text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
)
