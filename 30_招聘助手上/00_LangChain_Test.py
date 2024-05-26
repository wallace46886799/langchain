from langchain_openai import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import HumanMessage

text = "你好"
messages = [HumanMessage(content=text)]

# llm = ChatOpenAI(openai_api_key="xxx", openai_api_base="http://127.0.0.1:8000/v1")
# print(llm(messages))

embedding = OpenAIEmbeddings(openai_api_key="xxx", openai_api_base="http://127.0.0.1:8000/v1")
print(embedding.embed_documents(["你好"]))
