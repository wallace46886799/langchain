from loguru import logger

from langchain.callbacks import FileCallbackHandler
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量
logfile = "output.log"

logger.add(logfile, colorize=True, enqueue=True)
handler = FileCallbackHandler(logfile)

llm = ChatOpenAI()
prompt = PromptTemplate.from_template("1 + {number} = ")

# this chain will both print to stdout (because verbose=True) and write to 'output.log'
# if verbose=False, the FileCallbackHandler will still write to 'output.log'
chain = LLMChain(llm=llm, prompt=prompt, callbacks=[handler], verbose=True)
answer = chain.run(number=2)
logger.info(answer)
