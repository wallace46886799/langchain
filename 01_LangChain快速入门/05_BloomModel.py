import os
# 设置网络代理
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

# 通过.env管理huggingfacehub_api_token
from dotenv import load_dotenv
load_dotenv()

from langchain import HuggingFaceHub
llm = HuggingFaceHub(repo_id="bigscience/bloom-1b7")
resp = llm.predict("请给我的花店起个名")
print(resp)
