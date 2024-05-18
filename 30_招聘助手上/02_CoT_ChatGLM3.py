from dotenv import load_dotenv  # 用于加载环境变量
# 创建聊天模型
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import OutputFixingParser
from langchain.output_parsers import RetryWithErrorOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
# 定义我们想要接收的数据格式
from pydantic import BaseModel, Field


class CandidateDescription(BaseModel):
    jd_str: str = Field(description="岗位描述")
    resume_str: str = Field(description="候选人简历描述")
    description: str = Field(description="候选人匹配结果")
    reason: str = Field(description="候选人匹配结果的原因")


load_dotenv()  # 加载 .env 文件中的环境变量
llm = ChatOpenAI(temperature=0)

# 设定 AI 的角色和目标
role_template = """你是一个为金融科技软件公司工作的AI招聘助手, 你的目标是帮助客户根据他们的需要做出明智的决定"""

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

human_template = """岗位描述：{jd_str}，候选人简历：{resume_str}。对于这个岗位和候选人你有什么建议吗?
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
# 接收用户的询问，返回回答结果
output = llm(prompt)
output_content = output.content
print("-"*150)
print("原始输出：",output_content)
print("-"*150)

parsed_output = retry_parser.parse_with_prompt(output_content, prompt)

print("*"*150)
print("解析输出：",parsed_output)
print("*"*150)
