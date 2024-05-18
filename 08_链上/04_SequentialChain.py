'''欢迎来到LangChain实战课
https://time.geekbang.org/column/intro/100617601
作者 黄佳'''
# 设置OpenAI API密钥
# import os
# os.environ["OPENAI_API_KEY"] = 'Your Key'

from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量

# 导入所需要的库
# from langchain.llms import OpenAI
# from langchain.chains import LLMChain, SequentialChain
# from langchain.prompts import PromptTemplate

from langchain.chains import LLMChain,SequentialChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# 第一个LLMChain：生成鲜花的介绍
llm = ChatOpenAI(temperature=.7)
template = """
你是一个植物学家。给定花的名称和类型，你需要为这种花写一个200字左右的介绍。
花名: {name}
颜色: {color}
植物学家: 这是关于上述花的介绍:"""
prompt_template = PromptTemplate(
    input_variables=["name", "color"],
    template=template
)
introduction_chain = LLMChain(
    llm=llm,
    prompt=prompt_template,
    output_key="introduction"
)

# 第二个LLMChain：根据鲜花的介绍写出鲜花的评论
template = """
你是一位鲜花评论家。给定一种花的介绍，你需要为这种花写一篇200字左右的评论。
鲜花介绍:
{introduction}
花评人对上述花的评论:"""
prompt_template = PromptTemplate(
    input_variables=["introduction"],
    template=template
)
review_chain = LLMChain(
    llm=llm,
    prompt=prompt_template,
    output_key="review"
)

# 第三个LLMChain：根据鲜花的介绍和评论写出一篇自媒体的文案
template = """
你是一家花店的社交媒体经理。给定一种花的介绍和评论，你需要为这种花写一篇社交媒体的帖子，300字左右。
鲜花介绍:
{introduction}
花评人对上述花的评论:
{review}
社交媒体帖子:
"""
prompt_template = PromptTemplate(
    input_variables=["introduction", "review"],
    template=template
)
social_post_chain = LLMChain(
    llm=llm,
    prompt=prompt_template,
    output_key="social_post_text"
)

# 总的链：按顺序运行三个链
overall_chain = SequentialChain(
    chains=[introduction_chain, review_chain, social_post_chain],
    input_variables=["name", "color"],
    output_variables=["introduction", "review", "social_post_text"],
    verbose=True
)

# 运行链并打印结果
result = overall_chain.invoke({
    "name": "玫瑰",
    "color": "黑色"
})
from loguru import logger
logger.debug(result)
"""
2024-05-16 15:59:03.250 | DEBUG    | __main__:<module>:86 - {
'name': '玫瑰', 
'color': '黑色', 
'introduction': '玫瑰是一种非常美丽的花朵，其独特的黑色色调使其在花园中成为引人注目的存在。玫瑰属于蔷薇科，是一种多年生灌木植物。它们通常呈现出丰富的花瓣和迷人的花蕾形状，给人一种华丽而娇艳的感觉。\n\n黑色玫瑰是一种相对罕见的品种，其深邃的色彩给人一种神秘和浪漫的感觉。它们可以在花园中作为焦点或作为花束的一部分而使用。黑色玫瑰的花瓣通常是光滑而有光泽的，散发出一种淡淡的香气，让人陶醉其中。\n\n在栽培方面，玫瑰是一种相对容易生长的植物。它们喜欢阳光充足的环境，并且需要适当的灌溉和肥料。这些花朵通常在春季和夏季开花，为花园带来一片绚丽色彩。除了黑色，玫瑰还有各种各样的颜色和品种，每一种都有其独特的魅力。\n\n除了其美丽的外观，玫瑰在文化和象征意义上也有着重要的地位。它们被广泛用于庆祝和表达爱意。在不同的文化中，玫瑰被赋予了不同的意义，例如爱情、美丽、纯洁和神秘。\n\n总之，黑色玫瑰是一种迷人而神秘的花朵，它们的独特色彩和花朵形状使其成为花园中的亮点。无论是在花束中还是在花园中，玫瑰都能给人带来美丽和浪漫的感觉。对于热爱花卉的人们来说，玫瑰绝对是不可或缺的选择。', 
'review': '黑色玫瑰是一种非常特别的花朵，它的神秘和浪漫的感觉让人难以忘怀。它在花园中的存在就像是一颗黑色的珍珠，独特而引人注目。黑色的花瓣光滑而有光泽，给人一种高贵而优雅的感觉。花蕾形状迷人，每一朵花都散发着淡淡的香气，让人陶醉其中。\n\n玫瑰作为一种相对容易栽培的植物，它们需要充足的阳光、适量的水分和合适的肥料。在春季和夏季，它们的花朵盛开，为花园增添了一片绚丽的色彩。除了黑色，玫瑰还有许多其他颜色和品种，每一种都有其独特的魅力，使人们无法抗拒。\n\n除了其美丽的外观，玫瑰还具有丰富的文化和象征意义。它们是庆祝和表达爱意的常见选择。在不同的文化中，玫瑰被赋予了不同的含义，例如爱情、美丽、纯洁和神秘。无论是作为礼物还是装饰，玫瑰都能传递出美丽和浪漫的情感。\n\n总的来说，黑色玫瑰是一种迷人而神秘的花朵。它们的独特色彩和花朵形状使其成为花园中的亮点，无论是在花束中还是在花园中都能给人带来美丽和浪漫的感觉。对于热爱花卉的人们来说，黑色玫瑰绝对是不可或缺的选择。无论是在特殊场合还是日常生活中，它们都能给人带来喜悦和美好的回忆。', 
'social_post_text': '🌹✨ 令人陶醉的黑色玫瑰 ✨🌹\n\n大家好！今天我要向大家介绍一种非常特别的花朵——黑色玫瑰。这种玫瑰有着神秘和浪漫的感觉，无论是在花园中还是作为花束的一部分，都能给人带来美丽和浪漫的感觉。\n\n黑色玫瑰在花园中就像一颗黑色的珍珠，独特而引人注目。它们的花瓣光滑而有光泽，给人一种高贵而优雅的感觉。花蕾形状迷人，每一朵花都散发着淡淡的香气，让人陶醉其中。\n\n栽培黑色玫瑰相对容易，它们喜欢阳光充足的环境，并且需要适当的灌溉和肥料。在春季和夏季，它们的花朵盛开，为花园增添了一片绚丽的色彩。除了黑色，玫瑰还有许多其他颜色和品种，每一种都有其独特的魅力，使人们无法抗拒。\n\n除了其美丽的外观，玫瑰还具有丰富的文化和象征意义。它们是庆祝和表达爱意的常见选择。无论是在特殊场合还是日常生活中，玫瑰都能传递出美丽和浪漫的情感。在不同的文化中，玫瑰被赋予了不同的含义，例如爱情、美丽、纯洁和神秘。\n\n总的来说，黑色玫瑰是一种迷人而神秘的花朵。它们的独特色彩和花朵形状使其成为花园中的亮点。无论是在花束中还是在花园中，黑色玫瑰都能给人带来美丽和浪漫的感觉。对于热爱花卉的人们来说，黑色玫瑰绝对是不可或缺的选择。无论是送给自己还是给亲朋好友，它们都能给人带来喜悦和美好的回忆。\n\n如果你也迷恋黑色玫瑰，不妨在下方留言分享你对它的喜爱和美丽回忆吧！💖✨\n\n#黑色玫瑰 #迷人神秘 #美丽浪漫 #花卉爱好者'}
"""
