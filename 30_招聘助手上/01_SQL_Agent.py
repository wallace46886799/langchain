import os
from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量

from langchain.utilities import SQLDatabase
from langchain.llms import OpenAI
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType

# 连接到FlowerShop数据库
# db = SQLDatabase.from_uri("sqlite:///FlowerShop.db")
# url: jdbc:mysql://192.168.1.60:3306/sra?characterEncoding=UTF-8&useUnicode=true&useSSL=false&tinyInt1isBit=false&allowPublicKeyRetrieval=true&serverTimezone=Asia/Shanghai
# username: sra
# password: k$W6DVmfCP@VQEyH9yc
from urllib.parse import quote_plus as urlquote
def get_url():
    # user = os.getenv("MYSQL_USER", "dreamlabs")
    # password = os.getenv("MYSQL_PASSWORD", "frankbj828100")
    # host = os.getenv("MYSQL_HOST", "db")
    # db = os.getenv("MYSQL_DATABASE", "genius")
    user = os.getenv("MYSQL_USER", "sra")
    password = os.getenv("MYSQL_PASSWORD", "k$W6DVmfCP@VQEyH9yc")
    host = os.getenv("MYSQL_HOST", "192.168.1.60")
    db = os.getenv("MYSQL_DATABASE", "sra")
    return f"mysql+pymysql://{user}:{urlquote(password)}@{host}/{db}"

db_url = get_url()
db = SQLDatabase.from_uri(db_url)
# mysql+pymysql://user:pass@some_mysql_db_address/db_name
llm = ChatOpenAI(temperature=0, verbose=True)

# 创建SQL Agent
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=SQLDatabaseToolkit(db=db, llm=llm),
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

# 使用Agent执行SQL查询
questions = [
    "招聘助手会将筛选任务记录在表：delphi_recmt_task中，请在这张表里查询下招聘助手今天执行了多少条筛选任务？",
    "招聘助手会将候选人记录在表：recmt_candicates中，请在这张表中查询下招聘助手今天筛选了多少候选人？",
]

for question in questions:
    response = agent_executor.run(question)
    print(response)
