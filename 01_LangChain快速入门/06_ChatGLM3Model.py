import torch
print("Torch version:",torch.__version__)
print("Is CUDA enabled?",torch.cuda.is_available())

from transformers import AutoTokenizer, AutoModel
model_name = "THUDM/chatglm3-6b"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half().cuda()
model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half().float()
model = model.eval()
response, history = model.chat(tokenizer, "请给我的花店起个名", history=[])
from loguru import logger
logger.debug("HuggingFaceHub的Text模型：{}；返回的花店名称为：{}".format(model_name,response))
