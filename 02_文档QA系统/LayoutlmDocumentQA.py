from transformers import pipeline

document_qa = pipeline(model="impira/layoutlm-document-qa")
result = document_qa(
    image="https://huggingface.co/spaces/impira/docquery/resolve/2359223c1837a7587402bda0f2643382a6eefeab/invoice.png",
    question="What is the invoice date?",
)
print(result)

# https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.DocumentQuestionAnsweringPipeline
