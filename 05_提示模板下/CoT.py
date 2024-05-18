'''欢迎来到LangChain实战课
https://time.geekbang.org/column/intro/100617601
作者 黄佳'''
# 设置环境变量和API密钥
# import os
# os.environ["OPENAI_API_KEY"] = '你的OpenAI API Key'
from loguru import logger
from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量

# 创建聊天模型
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(temperature=0)

# 设定 AI 的角色和目标
role_template = "你是一个为花店电商公司工作的AI助手, 你的目标是帮助客户根据他们的喜好做出明智的决定"

# CoT 的关键部分，AI 解释推理过程，并加入一些先前的对话示例（Few-Shot Learning）
cot_template = """
作为一个为花店电商公司工作的AI助手，我的目标是帮助客户根据他们的喜好做出明智的决定。 

我会按部就班的思考，先理解客户的需求，然后考虑各种鲜花的涵义，最后根据这个需求，给出我的推荐。
同时，我也会向客户解释我这样推荐的原因。

示例 1:
  人类：我想找一种象征爱情的花。
  AI：首先，我理解你正在寻找一种可以象征爱情的花。在许多文化中，红玫瑰被视为爱情的象征，这是因为它们的红色通常与热情和浓烈的感情联系在一起。因此，考虑到这一点，我会推荐红玫瑰。红玫瑰不仅能够象征爱情，同时也可以传达出强烈的感情，这是你在寻找的。

示例 2:
  人类：我想要一些独特和奇特的花。
  AI：从你的需求中，我理解你想要的是独一无二和引人注目的花朵。兰花是一种非常独特并且颜色鲜艳的花，它们在世界上的许多地方都被视为奢侈品和美的象征。因此，我建议你考虑兰花。选择兰花可以满足你对独特和奇特的要求，而且，兰花的美丽和它们所代表的力量和奢侈也可能会吸引你。
"""
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
system_prompt_role = SystemMessagePromptTemplate.from_template(role_template)
system_prompt_cot = SystemMessagePromptTemplate.from_template(cot_template)

# 用户的询问
human_template = "{human_input}"
human_prompt = HumanMessagePromptTemplate.from_template(human_template)

# 将以上所有信息结合为一个聊天提示
chat_prompt = ChatPromptTemplate.from_messages([system_prompt_role, system_prompt_cot, human_prompt])

prompt = chat_prompt.format_prompt(human_input="我想为我的女朋友购买一些花。她喜欢粉色和紫色。你有什么建议吗?").to_messages()

# 接收用户的询问，返回回答结果
response = llm.invoke(prompt)
logger.debug(response.content)

# content='根据你女朋友喜欢粉色和紫色的喜好，我会推荐以下几种花给你：\n\n
# 1. **粉色康乃馨（Carnation）**：康乃馨是一种美丽且经典的花朵，粉色的康乃馨通常象征着母爱、友谊和善良。它们的花语也包括关怀和感激之情，适合表达对女朋友的关心和感激之情。\n\n
# 2. **紫色勿忘我（Forget-Me-Not）**：勿忘我是一种小巧可爱的花朵，紫色的勿忘我通常象征着真爱和忠诚。这种花也代表着永恒的爱和美好的回忆，适合表达对女朋友的真挚感情。\n\n
# 3. **粉色玫瑰（Pink Rose）**：粉色玫瑰是一种温柔和浪漫的花朵，粉色通常代表着温柔、善良和关怀。送粉色玫瑰可以表达对女朋友的温柔爱意和关怀之情。\n\n综上所述，你可以考虑选择粉色康乃馨、紫色勿忘我或粉色玫瑰作为送给女朋友的花束。这些花朵都能很好地符合她喜欢粉色和紫色的喜好，同时传达出你对她的关心和爱意。祝你们的感情更加美好！'
