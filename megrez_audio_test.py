import torch
from modelscope import AutoModelForCausalLM

path = "/root/workspace/mmllm/models/AI-ModelScope/Megrez-3B-Omni"  # Change this to the path of the model.

model = (
    AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
    )
    .eval()
    .cuda()
)

# Chat with text and image
# messages = [
#     {
#         "role": "user",
#         "content": {
#             "text": "Please describe the content of the image.",
#             "image": "./data/sample_image.jpg",
#         },
#     },
# ]
user_text = (
            f"使湧位填充方法，以01111110尉濰首flag，数据为011011111111111111110010，求问傳送时要添加几个0____\n"
            f"Options: A: 1 B: 2 C: 3 D: 4\n"
            "Please answer with A/B/C/D only."
        )
# Chat 
# messages = [
#     {
#         "role":"system",
#         "content":"You are a professional assistant. Only answer with the letter option (A/B/C/D). Do not include explanations or other content."
#     },
#     {
#         "role": "user",
#         "content": {
#             "text": user_text,
#             "audio": "data/ceval/audio/ceval_20/00001.mp3",
#         },
#     },
# ]

messages = [
    {
        "role":"system",
        "content":"You are a professional assistant. Only answer with the letter option (A/B/C/D). Do not include explanations or other content."
    },
    {
        "role": "user",
        "content": {
            "text": "Please discribe the content of the image.",
            "image": "data/ceval/image/ceval_20/00001.png",
        },
    },
]

MAX_NEW_TOKENS = 100
response = model.chat(
    messages,
    sampling=False,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=0,
)
print(response)