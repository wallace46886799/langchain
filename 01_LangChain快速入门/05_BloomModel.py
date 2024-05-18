import os
# 设置网络代理
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

# 通过.env管理huggingfacehub_api_token
from dotenv import load_dotenv
load_dotenv()

from langchain import HuggingFaceHub
model_name = "bigscience/bloom-1b7"
llm = HuggingFaceHub(repo_id=model_name)
response = llm.predict("请给我的花店起个名")
from loguru import logger
logger.debug("HuggingFaceHub的Text模型：{}返回的花店名称为：{}".format(model_name,response))
