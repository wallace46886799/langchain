'''欢迎来到LangChain实战课
https://time.geekbang.org/column/intro/100617601
作者 黄佳'''
# 设置OpenAI和SERPAPI的API密钥
import os
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.llms import OpenAI
from langchain import SerpAPIWrapper
from langchain.agents.tools import Tool
from langchain import LLMMathChain
from langchain.tools.base import BaseTool
# 创建聊天模型
from langchain.output_parsers import RetryWithErrorOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
# 定义我们想要接收的数据格式
from pydantic import BaseModel, Field

# os.environ["OPENAI_API_KEY"] = 'Your Key'
# os.environ["SERPAPI_API_KEY"] = 'Your Key'
from dotenv import load_dotenv  # 用于加载环境变量

load_dotenv()  # 加载 .env 文件中的环境变量
os.environ["SERPAPI_API_KEY"] = '068163c315bc7a57a8d7c5321f79367aaa594dd3580661c7abeda29f507e52ea'  # 068163c315bc7a57a8d7c5321f79367aaa594dd3580661c7abeda29f507e52ea


class WeatherTool(BaseTool):
    """这是一个调用第三方服务的获取某地区当天的天气情况"""

    name = "获取天气工具"
    description = (
        "这是一个调用第三方服务的获取某地区天气情况"
    )

    def _run(
            self,
            city: str,
            run_manager=None,
    ) -> str:
        """Use the tool."""
        """ 实现具体逻辑
        1. 本地实现逻辑：不需要跟第三方交互
        2. 通过与第三方交互，获的结果，请求方式或http或client请求
        """
        return "sunny"

    async def _arun(
            self,
            city: str,
            run_manager = None,
    ) -> str:
        pass

class SchoolTool(BaseTool):
    """这是一个调用第三方服务的获取某地区当天的天气情况"""

    name = "判断学校是否在学校白名单中"
    description = (
        "这是一个调用第三方服务的判断学校是否在学校白名单中的工具"
    )

    def _run(
            self,
            city: str,
            run_manager=None,
    ) -> str:
        """Use the tool."""
        """ 实现具体逻辑
        1. 本地实现逻辑：不需要跟第三方交互
        2. 通过与第三方交互获的结果，请求方式或http或client请求
        """
        return "学校在白名单中"

    async def _arun(
            self,
            city: str,
            run_manager = None,
    ) -> str:
        pass



class MajorTool(BaseTool):
    """这是一个调用第三方服务的获取某地区当天的天气情况"""

    name = "判断专业是否在专业白名单中"
    description = (
        "这是一个调用第三方服务的判断专业是否在专业白名单中的工具"
    )

    def _run(
            self,
            city: str,
            run_manager=None,
    ) -> str:
        """Use the tool."""
        """ 实现具体逻辑
        1. 本地实现逻辑：不需要跟第三方交互
        2. 通过与第三方交互获的结果，请求方式或http或client请求
        """
        return "专业在白名单中"

    async def _arun(
            self,
            city: str,
            run_manager = None,
    ) -> str:
        pass


search = SerpAPIWrapper()
llm = ChatOpenAI(temperature=0)
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

# Tool(
#     name="Search",
#     func=search.run,
#     description="useful for when you need to answer questions about current events"
# ),
# Tool(
#     name="Calculator",
#     func=llm_math_chain.run,
#     description="useful for when you need to answer questions about math"
# ),

tools = [
    Tool(
            name="School",
            func=search.run,
            description="这是一个调用第三方服务的判断学校是否在学校白名单中的工具"
        ),
    Tool(
            name="Major",
            func=search.run,
            description="这是一个调用第三方服务的判断专业是否在专业白名单中的工具"
        ),
]
model = llm
planner = load_chat_planner(model)
executor = load_agent_executor(model, tools, verbose=True)
agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)

class CandidateDescription(BaseModel):
    jd_str: str = Field(description="岗位描述")
    resume_str: str = Field(description="候选人简历描述")
    description: str = Field(description="候选人匹配结果")
    reason: str = Field(description="候选人匹配结果的原因")




# 设定 AI 的角色和目标
role_template = "你是一个为金融科技软件公司工作的AI招聘助手, 你的目标是帮助客户根据他们的需要做出明智的决定"

# CoT 的关键部分，AI解释推理过程，并加入一些先前的对话示例（Few-Shot Learning）
cot_template = """
作为一个为金融科技软件公司工作的AI招聘助手，我的目标是帮助客户根据他们的需要做出明智的决定。 

我会按部就班的思考，先理解客户的需求，然后考虑各种岗位的要求和候选人的简历内容，最后根据这个需求，给出我的推荐。
同时，我也会向客户解释我这样推荐的原因。

示例 1:
  人类：我想找一名Java高级工程师，本科以上学历，计算机相关专业毕业，工作经验5年以上。
  AI：首先，我理解你正在寻找一名Java高级工程师，本科以上学历，计算机相关专业毕业，工作经验5年以上。因此，考虑到这一点，我会推荐张三，他的简历如下：张三，男，北京林业大学，计算机科学与技术，2005年本科毕业，熟练掌握Java。张三毕业于北京林业大学，学历为本科，专业为计算机科学与技术，熟练掌握Java，2005年毕业至今已工作19年，这是你在寻找的候选人。

示例 2:
  人类：我想找一名Python高级工程师，本科以上学历，计算机相关专业毕业，工作经验5年以上。
  AI：首先，我理解你正在寻找一名Python高级工程师，本科以上学历，计算机相关专业毕业，工作经验5年以上。因此，考虑到这一点，我会推荐李四，他的简历如下：李四，男，北京林业大学，计算机科学与技术，2005年研究生毕业，熟练掌握Python。李四毕业于北京林业大学，学历为研究生，专业为计算机科学与技术，2005年毕业至今已工作19年，这是你在寻找的候选人。
"""

system_prompt_role = SystemMessagePromptTemplate.from_template(role_template)
system_prompt_cot = SystemMessagePromptTemplate.from_template(cot_template)

output_parser = PydanticOutputParser(pydantic_object=CandidateDescription)
# new_parser = OutputFixingParser.from_llm(parser=output_parser, llm=llm)
# output_parser = new_parser

retry_parser = RetryWithErrorOutputParser.from_llm(parser=output_parser, llm=llm)

# 获取输出格式指示
format_instructions = output_parser.get_format_instructions()
# 打印提示
print("输出格式：", format_instructions)

human_template = """岗位描述：{jd_str}，候选人简历：{resume_str}。对于这个候选人是否匹配这个岗位你有什么建议吗?
{format_instructions}"""

human_prompt = HumanMessagePromptTemplate.from_template(human_template,
                                      partial_variables={"format_instructions": format_instructions})

# 将以上所有信息结合为一个聊天提示
chat_prompt = ChatPromptTemplate.from_messages([system_prompt_role, system_prompt_cot, human_prompt])

jd_str = """岗位名称：Java高级工程师
            学历：本科以上学历
            专业：计算机相关专业毕业
            工作经验：10年以上"""

resume_str = """姓名：彭小波
                性别：男
                毕业学校：重庆大学
                专业：材料科学与工程
                毕业时间：2005年
                学历：专科
                其他描述：熟练掌握Java"""

prompt = chat_prompt.format_prompt(jd_str=jd_str,resume_str=resume_str).to_messages()

print(prompt)


output = agent.run(prompt)

print("*"*150)
print(output)
print("*"*150)
