from openai import OpenAI

client = ChatOpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1/",
)


# Chat completion API
completion = client.completions.create(
    model="gpt-3.5-turbo",
    prompt="感冒了怎么办",
)
print(completion)


