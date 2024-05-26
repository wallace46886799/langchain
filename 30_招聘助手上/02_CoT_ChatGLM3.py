from dotenv import load_dotenv  # 用于加载环境变量
# 创建聊天模型
from langchain_openai import ChatOpenAI
from langchain.output_parsers import OutputFixingParser
from langchain.output_parsers import RetryWithErrorOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
# 定义我们想要接收的数据格式
from pydantic.v1.main import BaseModel, Field


class CandidateDescription(BaseModel):
    jd_str: str = Field(description="岗位描述")
    resume_str: str = Field(description="候选人简历描述")
    description: str = Field(description="候选人匹配结果")
    reason: str = Field(description="候选人匹配结果的原因")


load_dotenv()  # 加载 .env 文件中的环境变量
llm = ChatOpenAI(temperature=0, model_name="chatglm3-6b")

# 设定 AI 的角色和目标
role_template = """作为一个为金融科技软件公司工作的AI招聘助手，我的目标是帮助客户根据他们的岗位需要做出明确的候选人是否通过筛选的决定"""

# CoT 的关键部分，AI解释推理过程，并加入一些先前的对话示例（Few-Shot Learning）
cot_template = """
作为一个为金融科技软件公司工作的AI招聘助手，我的目标是帮助客户根据他们的岗位需要做出明确的候选人是否通过筛选的决定。 

我会按部就班的思考，先理解客户提出的岗位需求，然后查看候选人简历并一一比对岗位要求和候选人简历，如有任何一项不满足岗位要求都视为不通过，最后给出我的明确建议：通过或者不通过。
同时，我也会向客户解释我做出建议的原因。

我的答复格式示例如下：

示例1：
建议：通过；原因：候选人各方面都满足岗位需求。

示例2：
建议：不通过；原因：岗位要求性别为男，而候选人性别为女。
"""

# 示例 1:
#   人类：我想找一名Java高级工程师，本科以上学历，计算机相关专业毕业，工作经验5年以上。
#   AI：首先，我理解你正在寻找一名Java高级工程师，本科以上学历，计算机相关专业毕业，工作经验5年以上。因此，考虑到岗位需求，我会推荐张三，他的简历如下：张三，男，北京林业大学，计算机科学与技术，2005年本科毕业，熟练掌握Java。张三毕业于北京林业大学，学历为本科，专业为计算机科学与技术，熟练掌握Java，2005年毕业至今已工作19年，这是你在寻找的候选人。
#
# 示例 2:
#   人类：我想找一名Python高级工程师，本科以上学历，计算机相关专业毕业，工作经验5年以上。
#   AI：首先，我理解你正在寻找一名Python高级工程师，本科以上学历，计算机相关专业毕业，工作经验5年以上。因此，考虑到岗位需求，我会推荐李四，他的简历如下：李四，男，北京林业大学，计算机科学与技术，2005年研究生毕业，熟练掌握Python。李四毕业于北京林业大学，学历为研究生，专业为计算机科学与技术，2005年毕业至今已工作19年，这是你在寻找的候选人。


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

human_template = """岗位要求：{jd_str}，候选人简历：{resume_str}。对于这个岗位和候选人你有什么建议吗?"""

human_prompt = HumanMessagePromptTemplate.from_template(human_template,
                                      )
# partial_variables = {"format_instructions": format_instructions}
# 将以上所有信息结合为一个聊天提示
chat_prompt = ChatPromptTemplate.from_messages([system_prompt_role, system_prompt_cot, human_prompt])

jd_str = """薪资：16000
全日制本科：是
学校：专科及以上
专业：计算机
服务公司：
工作经验：5年以上
工作状态：离职-随时到岗,在职-月内到岗,在职-考虑机会
在线状态：在线,刚刚活跃,今日活跃,3天内活跃,本周活跃,本月活跃
性别：男性
年龄：60岁以下"""

resume_str = """姓名：侯莉颖
期望薪资：面议
工作经验：3年
工作状态：离职-随时到岗
性别:女
年龄：24岁
其他描述：本人学习能力强能承受较大压力以及工作强度，熟练掌握Java开发基本技能，乐于挑战难题，乐于思考，希望能称为贵公司的一员，收获成长并实现自身更大的价值。"""

prompt = chat_prompt.format_prompt(jd_str=jd_str,resume_str=resume_str).to_messages()
# 接收用户的询问，返回回答结果
output = llm(prompt)
output_content = output.content
print("-"*150)
print("原始输出：",output)
# print("原始输出：",output_content)
print("-"*150)

# parsed_output = retry_parser.parse_with_prompt(output_content, prompt)
#
# print("*"*150)
# print("解析输出：",parsed_output)
# print("*"*150)
